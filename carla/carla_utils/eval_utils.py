from collections import deque
import time
import math
import queue
import cv2
import numpy as np
import torch
import carla
from carla_utils.route_utils import get_closest_waypoint_index, calculate_route_length, check_hazards, is_lane_safe, get_enhanced_command, command_to_branch_id
from carla_utils.visualization_utils import draw_full_diagnostic
from carla_utils.route_utils import set_traffic_lights_time
from carla_utils.carla_server import CarlaServer

import sys
sys.path.append("/home/eeafzal/carla_simulator/PythonAPI/carla")
from agents.navigation.global_route_planner import GlobalRoutePlanner

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def evaluate_in_carla(model,
                      epoch, 
                      device, 
                      client_port=3000, 
                      world=None,
                      eval_scenario="base",
                      n_vehicles=40,
                      n_walkers=40,
                      spawn_index=50,
                      dest_index=76,
                      eval_render_size=(1920, 1080),
                      fov=110,
                      use_aux_camera=False,
                      aux_camera_size=(1280, 720),
                      save_to_wandb=False):
    EVAL_RENDER_SIZE = eval_render_size
    FOV = fov
    AUX_CAMERA_SIZE = aux_camera_size
    STACK_FRAMES = 1

    model.eval()
    eval_video_storage_path = "eval_videos"
    video_name = f"eval_epoch_{epoch:02d}_{eval_scenario}_{spawn_index}-{dest_index}.mp4"
    video_name = f"{eval_video_storage_path}/{video_name}"
    
    '''
    t0 = time.time()
    carla_server = CarlaServer(rpc_port=client_port)
    carla_proc = carla_server.start()
    t1 = time.time()
    print(f"CARLA server startup time: {t1 - t0:.2f} seconds")
    if carla_proc is None:
        print("Failed to start CARLA server. Skipping eval...")
        return 0.0, 0.0, 0.0, 0
    '''
    
    # Metrics State
    total_cte = 0.0
    frame_count = 0
    collision_occurred = False
    off_road_occurred = False
    
    traffic_actors = []

    try:
        print(f"Beginning eval for epoch {epoch} scenario {eval_scenario}...")
        # for actor in world.get_actors().filter('vehicle.*'): actor.destroy()
        # for actor in world.get_actors().filter('sensor.*'): actor.destroy()
        '''
        client = carla.Client('localhost', client_port)
        t2 = time.time()
        print(f"CARLA client connection time: {t2 - t1:.2f} seconds")
        client.set_timeout(60.0)
        world = client.load_world('Town10HD_Opt')
        t3 = time.time()
        print(f"CARLA world load time: {t3 - t2:.2f} seconds")
        world = client.reload_world()
        t99 = time.time()
        print(f"CARLA world reload time: {t99 - t3:.2f} seconds")
        
        # Cleanup
        for actor in world.get_actors().filter('vehicle.*'): actor.destroy()
        for actor in world.get_actors().filter('sensor.*'): actor.destroy()
        
        # Physics Settings (Match Training)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        world.apply_settings(settings)

        if eval_scenario == "dynamic":
            weather = carla.WeatherParameters.ClearNoon
            for _ in range(10): world.tick()
            traffic_actors = spawn_traffic(world, client, n_vehicles, n_walkers)
        elif eval_scenario == "weather":
            weather = carla.WeatherParameters.HardRainNoon
            weather.fog_density = 10.0
            weather.wetness = 100.0
        elif eval_scenario == "base":
            weather = carla.WeatherParameters.ClearNoon
        else:
            print(f"Unknown EVAL_SCENARIO: {eval_scenario}, defaulting to 'base'")
            weather = carla.WeatherParameters.ClearNoon
        world.set_weather(weather)
        
        set_traffic_lights_time(world)
        '''
        map_data = world.get_map()
        grp = GlobalRoutePlanner(map_data, sampling_resolution=2.0)
        
        spawn_points = map_data.get_spawn_points()
        start_pose = spawn_points[spawn_index]
        dest_pose = spawn_points[dest_index]
        
        # Calculate Total Route Length for Completion %
        full_route = grp.trace_route(start_pose.location, dest_pose.location)
        total_route_length = calculate_route_length(full_route)
        
        # Spawn
        bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        vehicle = world.spawn_actor(bp, start_pose)
        
        # Sensors
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(EVAL_RENDER_SIZE[0]))
        cam_bp.set_attribute('image_size_y', str(EVAL_RENDER_SIZE[1]))
        cam_bp.set_attribute('fov', str(FOV))
        camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)

        if use_aux_camera:
            aux_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            aux_bp.set_attribute('image_size_x', str(AUX_CAMERA_SIZE[0]))
            aux_bp.set_attribute('image_size_y', str(AUX_CAMERA_SIZE[1]))
            aux_bp.set_attribute('fov', str(FOV))
            aux_camera = world.spawn_actor(aux_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
            # We won't use the aux camera for inference, but we can record it for diagnostics if needed.
            aux_q = queue.Queue()
            aux_camera.listen(aux_q.put)
            aux_video_name = f"aux_eval_epoch_{epoch:02d}.mp4"
            aux_video_out = cv2.VideoWriter(aux_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (AUX_CAMERA_SIZE[0], AUX_CAMERA_SIZE[1]))
        
        col_bp = world.get_blueprint_library().find('sensor.other.collision')
        collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        
        q = queue.Queue()
        camera.listen(q.put)
        
        col_q = queue.Queue()
        collision_sensor.listen(col_q.put)
        
        video_out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (EVAL_RENDER_SIZE[0], EVAL_RENDER_SIZE[1]))
        video_frames = []

        def clear_q(q):
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break

        clear_q(q)
        if use_aux_camera: clear_q(aux_q)

        # Inference State
        frame_stack = deque(maxlen=STACK_FRAMES)
        prev_action = torch.zeros(1, 3).to(device) # [Steer, Accel, Speed]
        current_route = list(full_route) # Copy
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        for _ in range(30): 
            world.tick()
            try: q.get(block=False, timeout=1.0)
            except queue.Empty: pass
            if use_aux_camera:
                try: aux_q.get(block=False, timeout=1.0)
                except queue.Empty: pass
        world.tick()

        vehicle.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=1))
        for _ in range(5): world.tick()
        vehicle.apply_control(carla.VehicleControl(throttle=1.0))
        for _ in range(10): world.tick()
        vehicle.apply_control(carla.VehicleControl(throttle=0.4, manual_gear_shift=False, hand_brake=False))
        clear_q(q)
        if use_aux_camera: clear_q(aux_q)
        for _ in range(50):
            world.tick()
            try: q.get(block=False, timeout=1.0)
            except queue.Empty: pass
            if use_aux_camera:
                try: aux_q.get(block=False, timeout=1.0)
                except queue.Empty: pass
        world.tick()

        print(f"Evaluating Epoch {epoch}, {eval_scenario}, Route {spawn_index}-{dest_index}...")
        stop_counter = 0

        full_route = grp.trace_route(start_pose.location, dest_pose.location)
        total_route_length = 0.0
        cumulative_distances = [0.0]

        for i in range(len(full_route) - 1):
            # Calculate 2D distance between waypoints
            loc1 = full_route[i][0].transform.location
            loc2 = full_route[i+1][0].transform.location
            # Use 2D distance (X, Y) as requested
            d = math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)
            total_route_length += d
            cumulative_distances.append(total_route_length)

        # Progress State
        max_reached_idx = 0
        
        for f in range(5000):
            ego_loc = vehicle.get_location()
            ego_yaw = vehicle.get_transform().rotation.yaw
            
            # Check Collision
            if not col_q.empty():
                print("Collision detected!")
                collision_occurred = True
                break
            
            carla_img = q.get(block=True, timeout=2.0)
            if use_aux_camera:
                aux_img = aux_q.get(block=True, timeout=2.0)
            
            closest_idx, dist_to_route = get_closest_waypoint_index(ego_loc, full_route)
            
            total_cte += dist_to_route
            frame_count += 1

            ego_wp = map_data.get_waypoint(ego_loc, project_to_road=False, lane_type=carla.LaneType.Any)
            if ego_wp:
                if ego_wp.lane_type in [carla.LaneType.Sidewalk]:
                    print(f"Failed: Went Off-Road ({ego_wp.lane_type})")
                    off_road_occurred = True
                    break
            
            # Failure Condition: Drifting too far
            if dist_to_route > 20.0:
                print(f"Failed: CTE too high ({dist_to_route:.2f}m)")
                break
                
            if len(current_route) == 0 or ego_loc.distance(dest_pose.location) < 5.0:
                print("Success: Destination Reached!")
                completion_ratio = 100.0
                current_progress_m = total_route_length
                break
            
            #Check if stuck (Low velocity and no progress for 200 consecutive frames and not stopped at a red light)
            v_vec = vehicle.get_velocity()
            speed_ms = math.sqrt(v_vec.x**2 + v_vec.y**2 + v_vec.z**2)
            if speed_ms < 0.5 and check_hazards(vehicle, world, map_data) is None:
                stop_counter += 1
                if stop_counter > 100:
                    print("Failed: Stuck for too long")
                    break
            else:
                stop_counter = 0

            array = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
            array = np.reshape(array, (EVAL_RENDER_SIZE[1], EVAL_RENDER_SIZE[0], 4))[:, :, :3]
            video_frame = array.copy() # cv2.resize(array.copy(), (960, 540), interpolation=cv2.INTER_AREA)
            if use_aux_camera:
                aux_array = np.frombuffer(aux_img.raw_data, dtype=np.uint8)
                aux_array = np.reshape(aux_array, (AUX_CAMERA_SIZE[1], AUX_CAMERA_SIZE[0], 4))[:, :, :3]
                aux_frame = aux_array.copy()
            '''
            ah, aw = array.shape[:2]
            side = min(ah, aw)
            start_x = (aw - side) // 2
            start_y = (ah - side) // 2
            array = array[start_y:start_y+side, start_x:start_x+side]
            array = array.copy()
            '''
            res = cv2.resize(array, (256, 256), interpolation=cv2.INTER_AREA)
            resized = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float().to(device) / 255.0
            
            if len(frame_stack)==0: 
                for _ in range(STACK_FRAMES): frame_stack.append(tensor)
            else: frame_stack.append(tensor)
            
            input_stack = torch.cat(list(frame_stack), dim=0).unsqueeze(0)
            input_stack = (input_stack.reshape(-1, 3, 256, 256) - mean) / std
            input_stack = input_stack.reshape(1, 3 * STACK_FRAMES, 256, 256)
            
            search_len = min(len(current_route), 20)
            closest_idx = 0
            min_dist = float('inf')

            for i in range(search_len):
                d = current_route[i][0].transform.location.distance(ego_loc)
                if d < min_dist:
                    min_dist = d
                    closest_idx = i

            if min_dist < 15.0: 
                current_route = current_route[closest_idx:]
            
            hazard_msg = check_hazards(vehicle, world, map_data)

            if hazard_msg: instruction = hazard_msg
            else: instruction = get_enhanced_command(current_route, ego_loc, ego_yaw)

            _, dist_to_center = get_closest_waypoint_index(ego_loc, current_route)
            is_mid_maneuver = dist_to_center > 0.5
            if not is_mid_maneuver:
                if instruction == "change lane left":
                    if not is_lane_safe(vehicle, world, map_data, 'left'):
                        instruction = "follow lane"
                elif instruction == "change lane right":
                    if not is_lane_safe(vehicle, world, map_data, 'right'):
                        instruction = "follow lane"

            branch_id = command_to_branch_id(instruction)

            with torch.no_grad():
                pred = model(input_stack, prev_action)
                pred = pred[0, branch_id]
            
            s = pred[0].item()
            accel = pred[1].item()
            
            # Convert Accel -> Throttle/Brake
            if accel > 0: t = accel; b = 0.0
            else: t = 0.0; b = -accel
            
            # Deadzones
            # if t < 0.1: t = 0.0
            if b < 0.1: b = 0.0
            
            # Speed Feedback
            v_vec = vehicle.get_velocity()
            speed_ms = math.sqrt(v_vec.x**2 + v_vec.y**2 + v_vec.z**2)
            prev_action = torch.tensor([s, accel, speed_ms]).unsqueeze(0).to(device)

            # Anti-Stall (Boost low throttle)
            # if speed_ms < 0.5 and t > 0.1: t = max(t, 0.6); b = 0.0
            if speed_ms < 0.1 and b < 0.1 and "stop" not in instruction:
                t = max(t, 0.3) # Force a kickstart

            vehicle.apply_control(carla.VehicleControl(steer=float(s), throttle=float(t), brake=float(b), manual_gear_shift=False))
            world.tick()

            ego_loc = vehicle.get_location()
        
            # Only look 20 waypoints ahead of where we last were
            search_start = max_reached_idx
            search_end = min(max_reached_idx + 20, len(full_route))
            
            best_dist = float('inf')
            best_idx = max_reached_idx

            for i in range(search_start, search_end):
                wp_loc = full_route[i][0].transform.location
                # 2D distance to waypoint
                d = math.sqrt((ego_loc.x - wp_loc.x)**2 + (ego_loc.y - wp_loc.y)**2)
                
                # We only count a waypoint as "reached" if we are reasonably close to it
                # (e.g., within 5 meters) to prevent jumping through buildings
                if d < best_dist and d < 5.0:
                    best_dist = d
                    best_idx = i

            # Update the furthest progress we've ever made
            if best_idx > max_reached_idx:
                max_reached_idx = best_idx

            # Calculate Completion based on pre-calculated route length
            current_progress_m = cumulative_distances[max_reached_idx]
            completion_ratio = (current_progress_m / total_route_length) * 100.0
            
            # cv2.putText(resized, f"CTE: {dist_to_route:.2f}m", (10, 20), 0, 0.5, (0, 255, 255), 1)
            video_frame = draw_full_diagnostic(video_frame, [s, t, b], cmd=instruction)
            video_out.write(video_frame)
            if save_to_wandb:
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                video_frame = np.transpose(video_frame, (2, 0, 1))
                video_frames.append(video_frame)
            if use_aux_camera:
                aux_frame = draw_full_diagnostic(aux_frame, [s, t, b], cmd=instruction)
                aux_video_out.write(aux_frame)
    
        avg_cte = total_cte / frame_count if frame_count > 0 else 0.0
        
        print(f"Eval Result {epoch} {eval_scenario} {spawn_index}-{dest_index} : Completion={completion_ratio:.2f}% | Avg CTE={avg_cte:.3f}m | Collision={collision_occurred} | Off-road={off_road_occurred}")
        
        return current_progress_m, total_route_length, total_cte, frame_count

    except Exception as e:
        print(f"Eval Error: {e}")
        return 0.0, 0.0, 0.0, 0

    finally:
        if 'video_out' in locals(): video_out.release()
        if use_aux_camera: aux_video_out.release()
        if 'camera' in locals(): camera.destroy()
        if 'aux_camera' in locals(): aux_camera.destroy()
        if 'collision_sensor' in locals(): collision_sensor.destroy()
        if 'vehicle' in locals(): vehicle.destroy()
        if 'traffic_actors' in locals():
            for actor in traffic_actors:
                if actor.is_alive:
                    actor.destroy()
        # carla_server.stop()
        model.train()
        if save_to_wandb and WANDB_AVAILABLE and len(video_frames) > 0:
            video_array = np.stack(video_frames)
            wandb.log({f"Epoch {epoch} {eval_scenario} {spawn_index}-{dest_index} eval": wandb.Video(video_array, fps=20, format="mp4")}, step=epoch)
            video_frames = []


def evaluate_in_carla_gated(model,
                      epoch, 
                      device, 
                      client_port=3000, 
                      world=None,
                      eval_scenario="base",
                      n_vehicles=40,
                      n_walkers=40,
                      spawn_index=50,
                      dest_index=76,
                      eval_render_size=(1920, 1080),
                      fov=110,
                      use_aux_camera=False,
                      aux_camera_size=(1280, 720),
                      save_to_wandb=False):
    EVAL_RENDER_SIZE = eval_render_size
    FOV = fov
    AUX_CAMERA_SIZE = aux_camera_size
    STACK_FRAMES = 1

    model.eval()
    eval_video_storage_path = "eval_videos"
    video_name = f"eval_epoch_{epoch:02d}_{eval_scenario}_{spawn_index}-{dest_index}.mp4"
    video_name = f"{eval_video_storage_path}/{video_name}"
    
    # Metrics State
    total_cte = 0.0
    frame_count = 0
    collision_occurred = False
    off_road_occurred = False
    
    traffic_actors = []

    try:
        for actor in world.get_actors().filter('vehicle.*'): actor.destroy()
        for actor in world.get_actors().filter('sensor.*'): actor.destroy()

        map_data = world.get_map()
        grp = GlobalRoutePlanner(map_data, sampling_resolution=2.0)
        
        spawn_points = map_data.get_spawn_points()
        start_pose = spawn_points[spawn_index]
        dest_pose = spawn_points[dest_index]
        
        # Calculate Total Route Length for Completion %
        full_route = grp.trace_route(start_pose.location, dest_pose.location)
        total_route_length = calculate_route_length(full_route)
        
        # Spawn
        bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        vehicle = world.spawn_actor(bp, start_pose)
        
        # Sensors
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(EVAL_RENDER_SIZE[0]))
        cam_bp.set_attribute('image_size_y', str(EVAL_RENDER_SIZE[1]))
        cam_bp.set_attribute('fov', str(FOV))
        camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)

        if use_aux_camera:
            aux_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            aux_bp.set_attribute('image_size_x', str(AUX_CAMERA_SIZE[0]))
            aux_bp.set_attribute('image_size_y', str(AUX_CAMERA_SIZE[1]))
            aux_bp.set_attribute('fov', str(FOV))
            aux_camera = world.spawn_actor(aux_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
            # We won't use the aux camera for inference, but we can record it for diagnostics if needed.
            aux_q = queue.Queue()
            aux_camera.listen(aux_q.put)
            aux_video_name = f"aux_eval_epoch_{epoch:02d}.mp4"
            aux_video_out = cv2.VideoWriter(aux_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (AUX_CAMERA_SIZE[0], AUX_CAMERA_SIZE[1]))
        
        col_bp = world.get_blueprint_library().find('sensor.other.collision')
        collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        
        q = queue.Queue()
        camera.listen(q.put)
        
        col_q = queue.Queue()
        collision_sensor.listen(col_q.put)
        
        video_out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (EVAL_RENDER_SIZE[0], EVAL_RENDER_SIZE[1]))
        video_frames = []

        def clear_q(q):
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break

        clear_q(q)
        if use_aux_camera: clear_q(aux_q)

        # Inference State
        frame_stack = deque(maxlen=STACK_FRAMES)
        prev_action = torch.zeros(1, 3).to(device) # [Steer, Accel, Speed]
        current_route = list(full_route) # Copy
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        for _ in range(30): 
            world.tick()
            try: q.get(block=False, timeout=1.0)
            except queue.Empty: pass
            if use_aux_camera:
                try: aux_q.get(block=False, timeout=1.0)
                except queue.Empty: pass
        world.tick()

        vehicle.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=1))
        for _ in range(5): world.tick()
        vehicle.apply_control(carla.VehicleControl(throttle=1.0))
        for _ in range(10): world.tick()
        vehicle.apply_control(carla.VehicleControl(throttle=0.4, manual_gear_shift=False, hand_brake=False))
        clear_q(q)
        if use_aux_camera: clear_q(aux_q)
        for _ in range(50):
            world.tick()
            try: q.get(block=False, timeout=1.0)
            except queue.Empty: pass
            if use_aux_camera:
                try: aux_q.get(block=False, timeout=1.0)
                except queue.Empty: pass
        world.tick()

        print(f"Evaluating Epoch {epoch}, {eval_scenario}, Route {spawn_index}-{dest_index}...")
        stop_counter = 0

        full_route = grp.trace_route(start_pose.location, dest_pose.location)
        total_route_length = 0.0
        cumulative_distances = [0.0]

        for i in range(len(full_route) - 1):
            # Calculate 2D distance between waypoints
            loc1 = full_route[i][0].transform.location
            loc2 = full_route[i+1][0].transform.location
            # Use 2D distance (X, Y) as requested
            d = math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)
            total_route_length += d
            cumulative_distances.append(total_route_length)

        # Progress State
        max_reached_idx = 0
        
        for f in range(5000):
            ego_loc = vehicle.get_location()
            ego_yaw = vehicle.get_transform().rotation.yaw
            
            # Check Collision
            if not col_q.empty():
                print("Collision detected!")
                collision_occurred = True
                break
            
            carla_img = q.get(block=True, timeout=2.0)
            if use_aux_camera:
                aux_img = aux_q.get(block=True, timeout=2.0)
            
            closest_idx, dist_to_route = get_closest_waypoint_index(ego_loc, full_route)
            
            total_cte += dist_to_route
            frame_count += 1

            ego_wp = map_data.get_waypoint(ego_loc, project_to_road=False, lane_type=carla.LaneType.Any)
            if ego_wp:
                if ego_wp.lane_type in [carla.LaneType.Sidewalk]:
                    print(f"Failed: Went Off-Road ({ego_wp.lane_type})")
                    off_road_occurred = True
                    break
            
            # Failure Condition: Drifting too far
            if dist_to_route > 20.0:
                print(f"Failed: CTE too high ({dist_to_route:.2f}m)")
                break
                
            if len(current_route) == 0 or ego_loc.distance(dest_pose.location) < 5.0:
                print("Success: Destination Reached!")
                completion_ratio = 100.0
                current_progress_m = total_route_length
                break
            
            #Check if stuck (Low velocity and no progress for 200 consecutive frames and not stopped at a red light)
            v_vec = vehicle.get_velocity()
            speed_ms = math.sqrt(v_vec.x**2 + v_vec.y**2 + v_vec.z**2)
            if speed_ms < 0.5 and check_hazards(vehicle, world, map_data) is None:
                stop_counter += 1
                if stop_counter > 100:
                    print("Failed: Stuck for too long")
                    break
            else:
                stop_counter = 0

            array = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
            array = np.reshape(array, (EVAL_RENDER_SIZE[1], EVAL_RENDER_SIZE[0], 4))[:, :, :3]
            video_frame = array.copy() # cv2.resize(array.copy(), (960, 540), interpolation=cv2.INTER_AREA)
            if use_aux_camera:
                aux_array = np.frombuffer(aux_img.raw_data, dtype=np.uint8)
                aux_array = np.reshape(aux_array, (AUX_CAMERA_SIZE[1], AUX_CAMERA_SIZE[0], 4))[:, :, :3]
                aux_frame = aux_array.copy()
            res = cv2.resize(array, (256, 256), interpolation=cv2.INTER_AREA)
            resized = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float().to(device) / 255.0
            
            if len(frame_stack)==0: 
                for _ in range(STACK_FRAMES): frame_stack.append(tensor)
            else: frame_stack.append(tensor)
            
            input_stack = torch.cat(list(frame_stack), dim=0).unsqueeze(0)
            input_stack = (input_stack.reshape(-1, 3, 256, 256) - mean) / std
            input_stack = input_stack.reshape(1, 3 * STACK_FRAMES, 256, 256)
            
            search_len = min(len(current_route), 20)
            closest_idx = 0
            min_dist = float('inf')

            for i in range(search_len):
                d = current_route[i][0].transform.location.distance(ego_loc)
                if d < min_dist:
                    min_dist = d
                    closest_idx = i

            if min_dist < 15.0: 
                current_route = current_route[closest_idx:]
            
            hazard_msg = check_hazards(vehicle, world, map_data)

            if hazard_msg: instruction = hazard_msg
            else: instruction = get_enhanced_command(current_route, ego_loc, ego_yaw)

            _, dist_to_center = get_closest_waypoint_index(ego_loc, current_route)
            is_mid_maneuver = dist_to_center > 0.5
            if not is_mid_maneuver:
                if instruction == "change lane left":
                    if not is_lane_safe(vehicle, world, map_data, 'left'):
                        instruction = "follow lane"
                elif instruction == "change lane right":
                    if not is_lane_safe(vehicle, world, map_data, 'right'):
                        instruction = "follow lane"

            branch_id = command_to_branch_id(instruction)
            task_vec = torch.zeros(1, 6, dtype=torch.float32).to(device)
            task_vec[0, branch_id] = 1.0

            with torch.no_grad():
                pred, _ = model(input_stack, prev_action, task_vec, hard_masking=True)
                pred = pred[0, branch_id]
            
            s = pred[0].item()
            accel = pred[1].item()
            
            # Convert Accel -> Throttle/Brake
            if accel > 0: t = accel; b = 0.0
            else: t = 0.0; b = -accel
            
            # Deadzones
            # if t < 0.1: t = 0.0
            if b < 0.1: b = 0.0
            
            # Speed Feedback
            v_vec = vehicle.get_velocity()
            speed_ms = math.sqrt(v_vec.x**2 + v_vec.y**2 + v_vec.z**2)
            prev_action = torch.tensor([s, accel, speed_ms]).unsqueeze(0).to(device)

            # Anti-Stall (Boost low throttle)
            # if speed_ms < 0.5 and t > 0.1: t = max(t, 0.6); b = 0.0
            if speed_ms < 0.1 and b < 0.1 and "stop" not in instruction:
                t = max(t, 0.3) # Force a kickstart

            vehicle.apply_control(carla.VehicleControl(steer=float(s), throttle=float(t), brake=float(b), manual_gear_shift=False))
            world.tick()

            ego_loc = vehicle.get_location()
        
            # Only look 20 waypoints ahead of where we last were
            search_start = max_reached_idx
            search_end = min(max_reached_idx + 20, len(full_route))
            
            best_dist = float('inf')
            best_idx = max_reached_idx

            for i in range(search_start, search_end):
                wp_loc = full_route[i][0].transform.location
                # 2D distance to waypoint
                d = math.sqrt((ego_loc.x - wp_loc.x)**2 + (ego_loc.y - wp_loc.y)**2)
                
                # We only count a waypoint as "reached" if we are reasonably close to it
                # (e.g., within 5 meters) to prevent jumping through buildings
                if d < best_dist and d < 5.0:
                    best_dist = d
                    best_idx = i

            # Update the furthest progress we've ever made
            if best_idx > max_reached_idx:
                max_reached_idx = best_idx

            # Calculate Completion based on pre-calculated route length
            current_progress_m = cumulative_distances[max_reached_idx]
            completion_ratio = (current_progress_m / total_route_length) * 100.0
            
            # cv2.putText(resized, f"CTE: {dist_to_route:.2f}m", (10, 20), 0, 0.5, (0, 255, 255), 1)
            video_frame = draw_full_diagnostic(video_frame, [s, t, b], cmd=instruction)
            video_out.write(video_frame)
            if save_to_wandb:
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                video_frame = np.transpose(video_frame, (2, 0, 1))
                video_frames.append(video_frame)
            if use_aux_camera:
                aux_frame = draw_full_diagnostic(aux_frame, [s, t, b], cmd=instruction)
                aux_video_out.write(aux_frame)
    
        avg_cte = total_cte / frame_count if frame_count > 0 else 0.0
        
        print(f"Eval Result {epoch} {eval_scenario} {spawn_index}-{dest_index} : Completion={completion_ratio:.2f}% | Avg CTE={avg_cte:.3f}m | Collision={collision_occurred} | Off-road={off_road_occurred}")
        
        return current_progress_m, total_route_length, total_cte, frame_count

    except Exception as e:
        print(f"Eval Error: {e}")
        return 0.0, 0.0, 0.0, 0

    finally:
        if 'video_out' in locals(): video_out.release()
        if use_aux_camera: aux_video_out.release()
        if 'camera' in locals(): camera.destroy()
        if 'aux_camera' in locals(): aux_camera.destroy()
        if 'collision_sensor' in locals(): collision_sensor.destroy()
        if 'vehicle' in locals(): vehicle.destroy()
        if 'traffic_actors' in locals():
            for actor in traffic_actors:
                if actor.is_alive:
                    actor.destroy()
        model.train()
        if save_to_wandb and WANDB_AVAILABLE and len(video_frames) > 0:
            video_array = np.stack(video_frames)
            wandb.log({f"Epoch {epoch} {eval_scenario} {spawn_index}-{dest_index} eval": wandb.Video(video_array, fps=20, format="mp4")}, step=epoch)
            video_frames = []
