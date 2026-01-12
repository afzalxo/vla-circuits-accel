import carla
import random
import time
import numpy as np
import cv2
import os
import pandas as pd
import queue
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_DIR = "carla_vla_dataset_hd"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_PATH = os.path.join(OUTPUT_DIR, "data.csv")
TOTAL_FRAMES = 20000
IMAGE_SIZE = 224
FOV = 90

def bin_action(control):
    steer = control.steer
    throttle = control.throttle
    brake = control.brake
    if brake > 0.1 or throttle < 0.05: return 0 # Stop
    if steer < -0.4: return 7 # Hard Left
    if steer > 0.4:  return 8 # Hard Right
    if steer < -0.1: return 4 if throttle > 0.6 else 3
    elif steer > 0.1: return 6 if throttle > 0.6 else 5
    else: return 2 if throttle > 0.6 else 1

def kickstart_vehicle(world, vehicle, image_queue):
    """
    Manually drives the vehicle forward for a few frames to wake up physics
    and get it moving before engaging Autopilot.
    """
    vehicle.set_autopilot(False)
    
    # Drive forward hard for 20 ticks (2 seconds sim time)
    for _ in range(20):
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0))
        world.tick()
        # Flush queue
        try:
            while not image_queue.empty(): image_queue.get(block=False)
        except queue.Empty: pass

def main():
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    
    # 1. Load Map
    print("Loading Town10HD...")
    world = client.load_world('Town10HD')
    
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # 2. Setup Settings
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)

    actor_list = []
    data_records = []

    try:
        # 3. Spawn Vehicle
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        
        vehicle = None
        while vehicle is None:
            spawn_point = random.choice(spawn_points)
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

        # 4. Spawn Camera
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(IMAGE_SIZE))
        cam_bp.set_attribute('image_size_y', str(IMAGE_SIZE))
        cam_bp.set_attribute('fov', str(FOV))
        
        spawn_point = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
        actor_list.append(camera)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        # --- WARMUP PHASE ---
        print("Kickstarting vehicle...")
        kickstart_vehicle(world, vehicle, image_queue)
        
        print("Engaging Autopilot...")
        vehicle.set_autopilot(True, tm.get_port())
        tm.ignore_lights_percentage(vehicle, 100.0)
        tm.ignore_signs_percentage(vehicle, 100.0)
        tm.vehicle_percentage_speed_difference(vehicle, -20.0)

        # Wait for stabilization
        warmup_tick = 0
        while True:
            world.tick()
            try:
                while not image_queue.empty(): image_queue.get(block=False)
            except queue.Empty: pass
            
            v = vehicle.get_velocity()
            speed = (v.x**2 + v.y**2 + v.z**2)**0.5
            
            if speed > 1.0:
                print(f"Vehicle stabilized at {speed:.2f} m/s. Starting collection.")
                break
            
            warmup_tick += 1
            if warmup_tick > 50:
                print("Stuck. Respawning and Kicking...")
                vehicle.set_transform(random.choice(spawn_points))
                world.tick() # Settle
                kickstart_vehicle(world, vehicle, image_queue)
                vehicle.set_autopilot(True, tm.get_port())
                warmup_tick = 0

        # 5. Collection Loop
        for frame_idx in tqdm(range(TOTAL_FRAMES)):
            world.tick()
            
            try:
                carla_image = image_queue.get(block=True, timeout=2.0)
            except queue.Empty:
                print("Timeout waiting for image.")
                continue

            control = vehicle.get_control()
            
            # Simple Instruction Logic for Town01
            # Town01 is mostly straight roads and T-junctions
            instruction = "follow the lane"
            if abs(control.steer) > 0.15:
                instruction = "turn left" if control.steer < 0 else "turn right"
            elif control.brake > 0.1:
                instruction = "stop"

            # Save Image
            array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (IMAGE_SIZE, IMAGE_SIZE, 4))
            array = array[:, :, :3]
            
            filename = f"{frame_idx:06d}.png"
            cv2.imwrite(os.path.join(IMG_DIR, filename), array)
            
            data_records.append({
                "image_path": filename,
                "instruction": instruction,
                "action": bin_action(control),
                "steer": control.steer,
                "throttle": control.throttle,
                "brake": control.brake
            })
            
            # Periodic Reset Check
            v = vehicle.get_velocity()
            speed = (v.x**2 + v.y**2 + v.z**2)**0.5
            
            if speed < 0.1 and frame_idx % 100 == 0:
                # Stuck? Respawn and Kickstart
                vehicle.set_transform(random.choice(spawn_points))
                world.tick()
                kickstart_vehicle(world, vehicle, image_queue)
                vehicle.set_autopilot(True, tm.get_port())

    finally:
        print("Cleaning up...")
        for actor in actor_list:
            if actor.is_alive: actor.destroy()
        
        df = pd.DataFrame(data_records)
        df.to_csv(CSV_PATH, index=False)
        print(f"Saved {len(df)} records.")
        
        # Restore settings
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == "__main__":
    main()
