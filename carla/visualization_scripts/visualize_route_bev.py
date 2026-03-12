import argparse
import sys
import carla
import cv2
import numpy as np
import queue
import time
from carla_utils.carla_server import CarlaServer
from carla_utils.route_utils import get_enhanced_command

sys.path.append("/home/eeafzal/carla_simulator/PythonAPI/carla")
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption


parser = argparse.ArgumentParser(description="CARLA Route Visualization")
parser.add_argument("--start_idx", type=int, default=50, help="Index of the starting spawn point")
parser.add_argument("--dest_idx", type=int, default=76, help="Index of the destination spawn point")
args = parser.parse_args()

IMAGE_RES = 1024
FOV = 100

def main():
    start_idx = args.start_idx
    dest_idx = args.dest_idx
    output_dir = "route_visualizations"
    output_filename = f"route_{start_idx}-{dest_idx}.png"
    output_filename = f"{output_dir}/{output_filename}"
    print("Connecting to CARLA...")
    carla_server = CarlaServer()
    carla_proc = carla_server.start()
    if carla_proc is None:
        print("Failed to start CARLA server. Exiting.")
        return

    client = carla.Client('localhost', 3000)
    client.set_timeout(20.0)
    world = client.load_world('Town10HD_Opt')
    map_data = world.get_map()
    
    world.set_weather(carla.WeatherParameters.ClearNoon)
    
    grp = GlobalRoutePlanner(map_data, sampling_resolution=2.0)
    
    spawn_points = map_data.get_spawn_points()
    start_pose = spawn_points[start_idx]
    dest_pose = spawn_points[dest_idx]
    
    print(f"Calculating route from Spawn {start_idx} to Spawn {dest_idx}...")
    route = grp.trace_route(start_pose.location, dest_pose.location)
    current_route = list(route)
    
    # 3. Visualization Logic
    debug = world.debug
    
    # Track min/max coordinates to center the camera later
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    # Draw Start (Green) and End (Red)
    debug.draw_box(carla.BoundingBox(start_pose.location, carla.Vector3D(2,2,2)), 
                   start_pose.rotation, 0.5, carla.Color(0, 255, 0), 0)
    debug.draw_string(start_pose.location + carla.Location(z=2), "START", 
                      draw_shadow=True, color=carla.Color(0, 255, 0), life_time=20)

    debug.draw_box(carla.BoundingBox(dest_pose.location, carla.Vector3D(2,2,2)), 
                   dest_pose.rotation, 0.5, carla.Color(255, 0, 0), 0)
    debug.draw_string(dest_pose.location + carla.Location(z=2), "DEST", 
                      draw_shadow=True, color=carla.Color(255, 0, 0), life_time=20)

    print(f"Route length: {len(route)} waypoints")
    prev_instruction = ""
    
    for i, (wp, cmd) in enumerate(route):
        loc = wp.transform.location
        yaw = wp.transform.rotation.yaw
        
        # Update bounds
        min_x = min(min_x, loc.x); max_x = max(max_x, loc.x)
        min_y = min(min_y, loc.y); max_y = max(max_y, loc.y)

        search_len = min(len(current_route), 20)
        closest_idx = 0
        min_dist = float('inf')
        for j in range(search_len):
            dist = current_route[j][0].transform.location.distance(loc)
            if dist < min_dist:
                min_dist = dist
                closest_idx = j
        if min_dist < 15.0:
            current_route = current_route[closest_idx:]
        instruction = get_enhanced_command(current_route, loc, yaw)

        if instruction != prev_instruction:
            prev_instruction = instruction
            print(f"Waypoint {i}: {instruction}")

        if "follow lane" in instruction.lower():
            color = carla.Color(0, 0, 255) # Blue
            symbol = 'X'
            debug.draw_string(loc + carla.Location(z=1), "FOLLOW LANE", 
                              draw_shadow=True, color=color, life_time=20)
        elif "turn left" in instruction.lower():
            color = carla.Color(255, 255, 0) # Yellow
            symbol = 'X'
            debug.draw_string(loc + carla.Location(z=1), "TURN LEFT", 
                              draw_shadow=True, color=color, life_time=20)
        elif "turn right" in instruction.lower():
            color = carla.Color(255, 0, 0) # Red
            symbol = 'X'
            debug.draw_string(loc + carla.Location(z=1), "TURN RIGHT", 
                              draw_shadow=True, color=color, life_time=20)
        elif "go straight" in instruction.lower():
            color = carla.Color(0, 255, 255) # Cyan
            symbol = '^'
            debug.draw_string(loc + carla.Location(z=1), "GO STRAIGHT", 
                              draw_shadow=True, color=color, life_time=20)
        elif "change lane left" in instruction.lower():
            color = carla.Color(255, 0, 255) # Magenta
            symbol = 'X'
            debug.draw_string(loc + carla.Location(z=1), "CHANGE LANE LEFT", 
                              draw_shadow=True, color=color, life_time=20)
        elif "change lane right" in instruction.lower():
            color = carla.Color(0, 0, 0) # Deep Pink
            symbol = 'X'
            debug.draw_string(loc + carla.Location(z=1), "CHANGE LANE RIGHT", 
                              draw_shadow=True, color=color, life_time=20)
        else:
            color = carla.Color(128, 128, 128) # Gray for unknown
            symbol = '?'
            debug.draw_string(loc + carla.Location(z=1), instruction.upper(), 
                              draw_shadow=True, color=color, life_time=20)
        debug.draw_point(loc + carla.Location(z=0.5), 0.1, color, 20)

    # Calculate center of the route
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Calculate required height to see everything (approximate)
    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height)
    cam_z = max_dim * 1.2 # Go up 1.2x the width of the route
    if cam_z < 50: cam_z = 50 # Minimum height
    
    cam_loc = carla.Location(x=center_x, y=center_y, z=cam_z)
    cam_rot = carla.Rotation(pitch=-90, yaw=0, roll=0) # Look straight down
    
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(IMAGE_RES))
    cam_bp.set_attribute('image_size_y', str(IMAGE_RES))
    cam_bp.set_attribute('fov', str(FOV))
    
    camera = world.spawn_actor(cam_bp, carla.Transform(cam_loc, cam_rot))
    
    # Queue for retrieving image
    image_queue = queue.Queue()
    camera.listen(image_queue.put)
    
    # Tick a few times to let the renderer catch up with the debug lines
    for _ in range(5):
        world.tick()
        if not image_queue.empty(): image_queue.get()
        time.sleep(0.1)
        
    # Capture
    image = image_queue.get()
    
    # Save
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (IMAGE_RES, IMAGE_RES, 4))[:, :, :3]
    cv2.imwrite(output_filename, array)
    
    print(f"Saved BEV map to {output_filename}")
    
    # Cleanup
    camera.destroy()
    carla_server.stop()

if __name__ == "__main__":
    main()
