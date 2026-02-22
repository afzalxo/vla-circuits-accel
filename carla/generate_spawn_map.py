import carla
import cv2
import numpy as np
import queue
import time
import os

MAP_NAME = 'Town10HD_Opt'
OUTPUT_FILENAME = "Town10_Spawn_Map.png"
IMG_SIZE = 2048
FOV = 90
FONT_SCALE = 0.60
FONT_THICKNESS = 3
POINT_RADIUS = 4

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)

    # UE4 Coordinate System (X-Forward, Y-Right, Z-Up) 
    # to Standard Camera System (x-Right, y-Down, z-Forward)
    # When camera is Pitch=-90 (Looking Down):
    # UE4 X (North) -> Camera -Y (Up in image, but standard CV is Down)
    # UE4 Y (East)  -> Camera X (Right)
    # UE4 Z (Up)    -> Camera -Z (Behind lens)
    
    # Standard transformation for Camera Sensor in CARLA:
    # [y, -z, x]
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # Project 3D->2D using the camera matrix
    if point_camera[2] < 0: return None # Behind camera

    u = K[0, 0] * point_camera[0] / point_camera[2] + K[0, 2]
    v = K[1, 1] * point_camera[1] / point_camera[2] + K[1, 2]

    return (int(u), int(v))

def main():
    print("Connecting to CARLA...")
    client = carla.Client('localhost', 3050)
    client.set_timeout(20.0)
    
    print(f"Loading {MAP_NAME}...")
    world = client.load_world(MAP_NAME)
    
    weather = carla.WeatherParameters.ClearNoon
    weather.fog_density = 0.0
    weather.cloudiness = 0.0
    weather.precipitation = 0.0
    world.set_weather(weather)

    print("Calculating Map Bounds...")
    spawn_points = world.get_map().get_spawn_points()
    x_vals = [sp.location.x for sp in spawn_points]
    y_vals = [sp.location.y for sp in spawn_points]
    
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Determine Z height required to cover the area with 90 deg FOV
    width = (max_x - min_x) * 1.3 # Add 30% padding
    height = (max_y - min_y) * 1.3
    cam_z = max(width, height) / 2.0 * 1.1 # 1.1 factor for safety
    if cam_z < 200: cam_z = 200 # Minimum height

    print(f"Map Center: ({center_x:.1f}, {center_y:.1f}) | Camera Height: {cam_z:.1f}")

    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(IMG_SIZE))
    cam_bp.set_attribute('image_size_y', str(IMG_SIZE))
    cam_bp.set_attribute('fov', str(FOV))
    # Sensor tick time (0.0 to capture as fast as possible, but we wait for manual tick)
    cam_bp.set_attribute('sensor_tick', '0.0')

    loc = carla.Location(x=center_x, y=center_y, z=cam_z)
    rot = carla.Rotation(pitch=-90, yaw=0, roll=0) # Look Straight Down
    
    camera = world.spawn_actor(cam_bp, carla.Transform(loc, rot))
    
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    print("Capturing High-Res Map...")
    # Warmup to let auto-exposure settle (though we are high up)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    try:
        for _ in range(10): 
            world.tick()
            try:
                image_queue.get(block=True, timeout=1.0)
            except queue.Empty:
                pass

        world.tick()
        image_data = image_queue.get(block=True, timeout=2.0)
        
        array = np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (IMG_SIZE, IMG_SIZE, 4))[:, :, :3]
        
        canvas = array.copy()

        print(f"Labeling {len(spawn_points)} spawn points...")
        
        K = build_projection_matrix(IMG_SIZE, IMG_SIZE, FOV)
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        for i, sp in enumerate(spawn_points):
            loc = sp.location
            pt = get_image_point(loc, K, world_2_camera)
            
            if pt:
                cv2.circle(canvas, pt, POINT_RADIUS + 2, (0, 0, 0), -1) 
                cv2.circle(canvas, pt, POINT_RADIUS, (0, 255, 255), -1) 
                
                text = str(i)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]
                text_x = pt[0] + 10 # Offset slightly to right
                text_y = pt[1] + 10
                
                cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                            FONT_SCALE, (0, 0, 0), FONT_THICKNESS + 4)
                cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                            FONT_SCALE, (0, 0, 255), FONT_THICKNESS)

        print(f"Saving to {OUTPUT_FILENAME}...")
        cv2.imwrite(OUTPUT_FILENAME, canvas)
        print("Done!")

    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        camera.destroy()

if __name__ == "__main__":
    main()
