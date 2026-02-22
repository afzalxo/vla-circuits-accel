import sys
import os
import carla
import queue
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
from collections import deque
import random
import gc
import cv2
import math
from models.carla_model import CarlaVLAHW as CarlaVLA
# from models.carla_model import CarlaVLASmall as CarlaVLA

sys.path.append("/home/eeafzal/carla_simulator/PythonAPI/carla")
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption

BASE_DATASET_DIR = "carla_dataset_base"
WEATHER_DATASET_DIR = "carla_dataset_weather"
DYNAMIC_DATASET_DIR = "carla_dataset_dynamic"

CARLA_CLIENT_PORT = 3000
EVAL_SCENARIO = "weather"
N_VEHICLES = 40
N_WALKERS = 40

BATCH_SIZE = 256 
NUM_WORKERS = 8 # CPU workers for data augmentation
PREFETCH_FACTOR = 2

LEARNING_RATE = 2.0e-4
EPOCHS = 50
MAX_TOKEN_LEN = 5
STACK_FRAMES = 1
IMG_SIZE = 256
WEIGHT_DECAY = 1e-3

ACTION_DROPOUT_PROB = 0.00
FLIP_PROB = 0.0
CONTROL_SCALE = 1.0

EVAL_RENDER_SIZE = (512, 512)
FOV = 110

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

def command_to_branch_id(cmd):
    cmd = str(cmd).lower().strip()
    if "left" in cmd:
        if "change" in cmd: return 3 # Change Lane Left
        return 1 # Turn Left
    elif "right" in cmd:
        if "change" in cmd: return 4 # Change Lane Right
        return 2 # Turn Right
    elif "stop" in cmd or "red light" in cmd or "brake" in cmd:
        return 5 # Stop at Red Light, Brake for Vehicle/Walker
    else:
        # Follow lane
        return 0

def spawn_traffic(tm, world, client, n_vehicles, n_walkers):
    """Spawns NPCs and Walkers for the Dynamic Scenario."""
    print(f"Spawning Traffic: {n_vehicles} Vehicles, {n_walkers} Walkers...")
    actor_list = []
    vehicle_list = []
    walker_list = []
    
    '''
    # 1. Setup Traffic Manager
    tm = client.get_trafficmanager(8000)
    tm.set_random_device_seed(42)
    tm.set_synchronous_mode(True)
    tm.set_hybrid_physics_mode(False) 
    # tm.set_hybrid_physics_radius(70.0)
    tm.global_percentage_speed_difference(30.0)
    tm.set_global_distance_to_leading_vehicle(3)
    '''

    # 2. Spawn Vehicles
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter('vehicle.*')
    vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]
    spawn_points = world.get_map().get_spawn_points()
    spawn_points.pop(50) # Remove our spawn point from the list to avoid collision
    
    # Randomly shuffle spawn points
    random.shuffle(spawn_points)
    
    for i in range(min(n_vehicles, len(spawn_points))):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        
        # Set autopilot
        bp.set_attribute('role_name', 'autopilot')
        spawn_point = spawn_points[i]
        spawn_point.location.z += 0.3
        
        try:
            v = world.spawn_actor(bp, spawn_point)
            v.set_autopilot(True, tm.get_port())
            tm.ignore_lights_percentage(v, 0) # Obey lights
            tm.auto_lane_change(v, True)
            actor_list.append(v)
            vehicle_list.append(v)
        except:
            pass
            
    # 3. Spawn Walkers
    # Walkers need a controller + the actual walker actor
    walker_bp = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find('controller.ai.walker')
    
    spawn_points = []
    for _ in range(n_walkers):
        loc = world.get_random_location_from_navigation()
        if loc: spawn_points.append(carla.Transform(loc))
    
    for spawn_point in spawn_points:
        try:
            bp = random.choice(walker_bp)
            walker = world.spawn_actor(bp, spawn_point)
            controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            
            # Start walking
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(1.4 + random.random()) # Random speed
            
            actor_list.append(walker)
            actor_list.append(controller)
            walker_list.append(walker)
        except:
            pass

    print(f"Spawned {len(vehicle_list)} Vehicles and {len(walker_list)} Walkers.")
    return actor_list

def reset_vehicle(vehicle, map_data):
    """Teleports vehicle to a random valid spawn point and clears velocity."""
    spawn_points = map_data.get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle.set_transform(spawn_point)
    vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
    # Give the physics engine a moment to settle
    return spawn_point

def get_enhanced_command(route, vehicle_loc, vehicle_yaw):
    if not route: return "follow lane"
    
    # --- TUNING PARAMETERS ---
    # 1. Lane Change: Increased to 30m to catch the command before steering starts
    LANE_CHANGE_TRIGGER = 30.0 
    
    # 2. Turns: Keep at 15m (seems to work well for entry)
    TURN_TRIGGER = 15.0        
    
    # 3. Junction Entry: 10m
    JUNCTION_TRIGGER = 10.0    
    
    cumulative_dist = 0.0
    prev_loc = vehicle_loc
    
    # --- PHASE 1: SCAN AHEAD (The Plan) ---
    for i in range(min(len(route), 50)):
        wp, cmd = route[i]
        curr_loc = wp.transform.location
        cumulative_dist += curr_loc.distance(prev_loc)
        prev_loc = curr_loc
        
        # Stop scanning if we went too far
        if cumulative_dist > LANE_CHANGE_TRIGGER: break
        
        # PRIORITY 1: LANE CHANGES (Long range)
        if cmd == RoadOption.CHANGELANELEFT: return "change lane left"
        if cmd == RoadOption.CHANGELANERIGHT: return "change lane right"

        # PRIORITY 2: TURNS (Medium range)
        if cumulative_dist < TURN_TRIGGER:
            if cmd in [RoadOption.LEFT, RoadOption.RIGHT]:
                # Verify geometry (prevent noise)
                wp_yaw = wp.transform.rotation.yaw
                diff = wp_yaw - vehicle_yaw
                while diff > 180: diff -= 360
                while diff < -180: diff += 360
                
                if diff > 15: return "turn right"
                if diff < -15: return "turn left"

        # PRIORITY 3: EXPLICIT STRAIGHT (Short range)
        if cumulative_dist < JUNCTION_TRIGGER:
            if cmd == RoadOption.STRAIGHT:
                if wp.is_junction: return "go straight"

    # --- PHASE 2: GAP FILLER (The Reality) ---
    # If we are physically inside a junction, we must maintain the command
    # because the lookahead might only see "LaneFollow" for the exit.
    current_wp = route[0][0]
    
    if current_wp.is_junction:
        # Look ONLY 5 meters ahead. 
        # Reducing this from 10m -> 5m fixes the "Linger" issue.
        # We want to know what the road is doing *right now*, not in 10m.
        lookahead_dist = 0.0
        prev_loc = vehicle_loc
        
        for i in range(min(len(route), 10)):
            wp = route[i][0]
            lookahead_dist += wp.transform.location.distance(prev_loc)
            prev_loc = wp.transform.location
            
            if lookahead_dist > 5.0:
                # Calculate angle to this immediate future point
                wp_yaw = wp.transform.rotation.yaw
                diff = wp_yaw - vehicle_yaw
                while diff > 180: diff -= 360
                while diff < -180: diff += 360
                
                # Geometric Classification
                # Stricter thresholds to snap back to straight earlier
                if diff > 20: return "turn right"
                if diff < -20: return "turn left"
                
                # If angle is small (< 10), we are effectively going straight
                if abs(diff) < 10: return "go straight"
                
                # If we are between 10 and 20 degrees, we are transitioning.
                # Defaulting to "follow lane" here helps smooth the transition out of the turn.
                return "follow lane"
                
    # Default
    return "follow lane"

def check_hazards(vehicle, world, map_data):
    if vehicle.is_at_traffic_light():
        if vehicle.get_traffic_light().get_state() == carla.TrafficLightState.Red:
            return "stop at red light"

    # 2. Geometry Setup
    ego_trans = vehicle.get_transform()
    ego_loc = ego_trans.location
    ego_fwd = ego_trans.get_forward_vector()
    
    # 3. Get all relevant actors
    # (In a highly optimized simulation, you'd pass this list in, but this works)
    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')
    walkers = actors.filter('walker.pedestrian.*')
    
    # 4. Define Safety Distances
    VEHICLE_BRAKE_DIST = 8.0
    WALKER_BRAKE_DIST = 12.0
    LANE_WIDTH_TOLERANCE = 1.8 # ~Half a lane width (detects overlapping objects)

    def is_hazard(target_actor, brake_dist):
        if target_actor.id == vehicle.id: return False
        
        target_loc = target_actor.get_location()
        dist_vec = target_loc - ego_loc
        
        # Calculate Longitudinal Distance (Forward/Backward)
        # Dot product: Projects the distance vector onto the forward vector
        long_dist = dist_vec.x * ego_fwd.x + dist_vec.y * ego_fwd.y + dist_vec.z * ego_fwd.z
        
        if long_dist < 0: return False # Behind us
        if long_dist > brake_dist: return False # Too far away
        
        # Calculate Lateral Distance (Side-to-Side)
        # Using Pythagorean theorem: Lat^2 = Dist^2 - Long^2
        dist_sq = dist_vec.x**2 + dist_vec.y**2 + dist_vec.z**2
        lat_dist = math.sqrt(max(0, dist_sq - long_dist**2))
        
        # If the object is within our "Tube" width, it's a hazard
        return lat_dist < LANE_WIDTH_TOLERANCE

    # 5. Check Vehicles
    for target in vehicles:
        if is_hazard(target, VEHICLE_BRAKE_DIST):
            return "brake for vehicle"

    # 6. Check Pedestrians (New)
    for target in walkers:
        if is_hazard(target, WALKER_BRAKE_DIST):
            return "brake for pedestrian"
            
    return None


def is_lane_safe(vehicle, world, map_data, direction):
    """
    Checks if the target lane (left or right) is clear of vehicles.
    direction: 'left' or 'right'
    """
    ego_loc = vehicle.get_location()
    ego_wpt = map_data.get_waypoint(ego_loc)
    
    # 1. Get the Target Waypoint
    target_wpt = None
    if direction == 'left':
        target_wpt = ego_wpt.get_left_lane()
    else: # right
        target_wpt = ego_wpt.get_right_lane()
        
    # If no lane exists (e.g. sidewalk/wall), it's not safe
    if not target_wpt or target_wpt.lane_type != carla.LaneType.Driving:
        return False

    # 2. Define Safety Zone parameters
    # At 15km/h, we need about 8-10 meters clearance front and back
    SAFETY_DIST_FRONT = 10.0 
    SAFETY_DIST_BACK = 8.0
    LANE_WIDTH_CHECK = 1.5 # Half lane width

    # 3. Check all vehicles
    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')
    
    for target in vehicles:
        if target.id == vehicle.id: continue
        
        target_loc = target.get_location()
        
        # We check distance relative to the TARGET WAYPOINT, not our car.
        # This effectively checks "Is anyone in the spot I want to move to?"
        
        # Vector from Target Waypoint to Other Vehicle
        vec_diff = target_loc - target_wpt.transform.location
        
        # Project onto lane direction
        fwd = target_wpt.transform.get_forward_vector()
        long_dist = vec_diff.x * fwd.x + vec_diff.y * fwd.y
        
        # Check Longitudinal (Front/Back)
        if -SAFETY_DIST_BACK < long_dist < SAFETY_DIST_FRONT:
            # Check Lateral (Is it actually in that lane?)
            # Distance from the center of the target lane
            # We use 2D Euclidean distance minus the longitudinal component
            dist_sq = vec_diff.x**2 + vec_diff.y**2 + vec_diff.z**2
            lat_dist = math.sqrt(max(0, dist_sq - long_dist**2))
            
            if lat_dist < LANE_WIDTH_CHECK:
                # print(f"Lane blocked by {target.type_id} at {long_dist:.1f}m")
                return False # Unsafe

    return True


def draw_full_diagnostic(img_bgr, model, expert=None, cmd=None):
    """
    expert/model: [steer, throttle, brake]
    """
    h, w = img_bgr.shape[:2]
    overlay_h = 110
    if expert is None:
        overlay_h = overlay_h // 2
    
    # Semi-transparent background for UI
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, overlay_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img_bgr, 0.5, 0, img_bgr)

    # 1. Text Info
    if cmd is not None:
        cv2.putText(img_bgr, f"CMD: {cmd.upper()}", (10, 25), 0, 0.6, (0, 255, 255), 2)
    
    # 2. Steering Lines (Bottom)
    start_pt = (w // 2, h - 20)
    if expert is not None:
        exp_steer_end = (int(w // 2 + expert[0] * 150), h - 70)
    prd_steer_end = (int(w // 2 + model[0] * 150), h - 70)
    if expert is not None:
        cv2.line(img_bgr, start_pt, exp_steer_end, (0, 255, 0), 3) # Expert Green
    cv2.line(img_bgr, start_pt, prd_steer_end, (0, 255, 0), 1) # Model Red

    bar_h = 10
    bar_y_start = 20
    dist_between = 15

    # 3. Throttle/Brake Bars (Top Right)
    # Expert Bars (Green)
    if expert is not None:
        cv2.putText(img_bgr, "EXPERT", (w - 180, 45), 0, 0.4, (0, 255, 0), 1)
        cv2.rectangle(img_bgr, (w - 100, bar_y_start), (w - 100 + int(expert[1]*80), bar_y_start + bar_h), (0, 255, 0), -1)
        cv2.rectangle(img_bgr, (w - 100, bar_y_start + dist_between), (w - 100 + int(expert[2]*80), bar_y_start + dist_between + bar_h), (0, 100, 0), -1)
    if expert is not None:
        bar_y_start += bar_y_start

    # Model Bars (Red)
    # cv2.putText(img_bgr, "MODEL", (w - 180, 85), 0, 0.4, (0, 0, 255), 1)
    # Throttle
    cv2.rectangle(img_bgr, (w - 100, bar_y_start), (w - 100 + int(model[1]*80), bar_y_start + bar_h), (0, 0, 255), -1)
    # Brake
    cv2.rectangle(img_bgr, (w - 100, bar_y_start + dist_between), (w - 100 + int(model[2]*80), bar_y_start + dist_between + bar_h), (100, 0, 100), -1)

    # Labels TODO
    if expert is not None:
        cv2.putText(img_bgr, "THR", (w - 130, 50), 0, 0.3, (255, 255, 255), 1)
        cv2.putText(img_bgr, "BRK", (w - 130, 65), 0, 0.3, (255, 255, 255), 1)
    # cv2.putText(img_bgr, "THR", (w - 130, 90), 0, 0.3, (255, 255, 255), 1)
    # cv2.putText(img_bgr, "BRK", (w - 130, 105), 0, 0.3, (255, 255, 255), 1)
    cv2.putText(img_bgr, f"{model[1]:.2f}", (w - 130, bar_y_start + 7), 0, 0.3, (255, 255, 255), 1)
    cv2.putText(img_bgr, f"{model[2]:.2f}", (w - 130, bar_y_start + dist_between + 7), 0, 0.3, (255, 255, 255), 1)

    return img_bgr


class GPUAugmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img_stack, is_train=True):
        # Input: [B, 12, H, W] uint8 tensor
        x = img_stack.float() / 255.0
        
        b, c, h, w = x.shape
        x = x.reshape(-1, 3, h, w)
        
        if is_train:
            # Brightness
            if random.random() < 0.8:
                factor = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
                x = x * factor
            
            # Contrast
            if random.random() < 0.8:
                factor = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
                mean = x.mean(dim=(1, 2, 3), keepdim=True)
                x = (x - mean) * factor + mean
                
            x = torch.clamp(x, 0, 1)

        x = (x - self.mean) / self.std
        x = x.reshape(b, c, h, w)
        return x

# --- 2. MODEL (6-Layer GAP) ---
class SimpleTokenizer:
    def __init__(self, texts):
        word_counts = Counter()
        for text in texts: word_counts.update(str(text).lower().split())
        self.vocab = {word: i+1 for i, (word, _) in enumerate(word_counts.items())}
        self.vocab_size = len(self.vocab) + 1
    def encode(self, text):
        return torch.tensor([self.vocab.get(w, 0) for w in str(text).lower().split()], dtype=torch.long)

def get_closest_waypoint_index(vehicle_loc, route):
    min_dist = float('inf')
    closest_idx = 0
    for i, (wp, _) in enumerate(route):
        dist = wp.transform.location.distance(vehicle_loc)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    return closest_idx, min_dist

def calculate_route_length(route):
    length = 0.0
    for i in range(len(route) - 1):
        length += route[i][0].transform.location.distance(route[i+1][0].transform.location)
    return length

def evaluate_in_carla(model, client, epoch, tokenizer, device, save_to_wandb=False):
    model.eval()
    video_name = f"eval_epoch_{epoch:02d}.mp4"
    
    # Metrics State
    total_cte = 0.0
    frame_count = 0
    collision_occurred = False
    off_road_occurred = False
    
    traffic_actors = []

    try:
        client = carla.Client('localhost', CARLA_CLIENT_PORT)
        client.set_timeout(5.0)
        world = client.get_world()
        # world = client.reload_world()
        
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

        if EVAL_SCENARIO == "dynamic":
            tm = client.get_trafficmanager(8000)
            tm.set_synchronous_mode(True)
            tm.set_random_device_seed(42)
            tm.set_hybrid_physics_mode(False)
            tm.global_percentage_speed_difference(30.0)
            tm.set_global_distance_to_leading_vehicle(3)
            for _ in range(10): world.tick()
            traffic_actors = spawn_traffic(tm, world, client, N_VEHICLES, N_WALKERS)
        elif EVAL_SCENARIO == "weather":
            weather = carla.WeatherParameters.HardRainNoon
            weather.fog_density = 10.0
            weather.wetness = 100.0
            world.set_weather(weather)

        map_data = world.get_map()
        grp = GlobalRoutePlanner(map_data, sampling_resolution=2.0)
        
        spawn_points = map_data.get_spawn_points()
        # start_pose = spawn_points[50]
        # dest_pose = spawn_points[76]
        start_pose = spawn_points[50]
        dest_pose = spawn_points[14]
        
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
        world.tick()

        vehicle.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=1))
        for _ in range(5): world.tick()
        vehicle.apply_control(carla.VehicleControl(throttle=1.0))
        for _ in range(10): world.tick()
        vehicle.apply_control(carla.VehicleControl(throttle=0.4, manual_gear_shift=False, hand_brake=False))
        clear_q(q)
        for _ in range(50):
            world.tick()
            try: q.get(block=False, timeout=1.0)
            except queue.Empty: pass
        world.tick()

        print(f"Evaluating Epoch {epoch}...")
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
            
            closest_idx, dist_to_route = get_closest_waypoint_index(ego_loc, full_route)
            
            total_cte += dist_to_route
            frame_count += 1

            ego_wp = map_data.get_waypoint(ego_loc, project_to_road=False, lane_type=carla.LaneType.Any)
            if ego_wp:
                if ego_wp.lane_type in [carla.LaneType.Sidewalk]:  # , carla.LaneType.Shoulder]:  # Removing shoulder lane for now ;)
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
            video_frame = array.copy()
            ah, aw = array.shape[:2]
            side = min(ah, aw)
            start_x = (aw - side) // 2
            start_y = (ah - side) // 2
            array = array[start_y:start_y+side, start_x:start_x+side]
            array = array.copy()
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
            
            # Command
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

            tokens = tokenizer.encode(instruction).unsqueeze(0).to(device)
            token_list = tokens.tolist()[0]
            while len(token_list) < MAX_TOKEN_LEN: token_list.append(0)
            tokens = torch.tensor(token_list).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(input_stack, prev_action)
                pred = pred[0, branch_id]
            
            s = pred[0].item() / CONTROL_SCALE
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
    
        avg_cte = total_cte / frame_count if frame_count > 0 else 0.0
        
        print(f"EVAL RESULT: Completion={completion_ratio:.1f}% | Avg CTE={avg_cte:.3f}m | Collision={collision_occurred} | Off-road={off_road_occurred}")
        
        return completion_ratio, avg_cte

    except Exception as e:
        print(f"Eval Error: {e}")
        return 0.0, 100.0
    finally:
        video_out.release()
        '''
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        actors_to_destroy = []
        if 'camera' in locals() and camera.is_alive: actors_to_destroy.append(camera)
        if 'collision_sensor' in locals() and collision_sensor.is_alive: actors_to_destroy.append(collision_sensor)
        if 'vehicle' in locals() and vehicle.is_alive: actors_to_destroy.append(vehicle)
        
        if traffic_actors:
            # traffic_actors contains both walkers and controllers
            for actor in traffic_actors:
                if actor is not None and actor.is_alive:
                    actors_to_destroy.append(actor)

        if actors_to_destroy:
            client.apply_batch([carla.command.DestroyActor(x) for x in actors_to_destroy])
        print(f"Cleaned up {len(actors_to_destroy)} actors.")
        '''

        if 'camera' in locals(): camera.destroy()
        if 'collision_sensor' in locals(): collision_sensor.destroy()
        if 'vehicle' in locals(): vehicle.destroy()
        # Destroy pedestrians
        if 'traffic_actors' in locals():
            for actor in traffic_actors:
                if actor.is_alive:
                    actor.destroy()
        model.train()
        if save_to_wandb:
            import wandb
            video_array = np.stack(video_frames)
            wandb.log({f"Epoch {epoch} Eval": wandb.Video(video_array, fps=20, format="mp4")}, step=epoch)
            video_frames = []


class CarlaImageCache:
    def __init__(self, csv_path, img_dir):
        # print(f"Loading Metadata from {csv_path}...")
        def process_df(path, scenario_name, is_rainy, is_crowded):
            df = pd.read_csv(os.path.join(path, "data.csv"))
            # Prepend the folder path to the image filename
            df['image_path'] = os.path.join(path, "images") + "/" + df['image_path']
            # Add metadata tags for the Gating Network
            df['is_rainy'] = is_rainy
            df['is_crowded'] = is_crowded
            # Ensure trajectory_id is unique across the whole dataset
            # (Offset each by 1,000,000 to prevent overlap)
            offset = {"base": 0, "weather": 1000000, "dynamic": 2000000}
            df['trajectory_id'] += offset[scenario_name]
            return df

        df_base = process_df(BASE_DATASET_DIR, "base", 0, 0)
        df_weather = process_df(WEATHER_DATASET_DIR, "weather", 1, 0)
        df_dynamic = process_df(DYNAMIC_DATASET_DIR, "dynamic", 0, 1)
        self.data = pd.concat([df_base, df_weather, df_dynamic], ignore_index=True)
        # self.data = df_base

        '''
        self.data = pd.read_csv(csv_path)
        def get_cam_code(path):
            if '_c.png' in path: return 0
            if '_l.png' in path: return 1
            if '_r.png' in path: return 2
            return 0
        self.data['cam_code'] = self.data['image_path'].apply(get_cam_code)
        self.data['virtual_traj_id'] = self.data['trajectory_id'] * 10 + self.data['cam_code']
        self.data = self.data.sort_values(['virtual_traj_id', 'image_path']).reset_index(drop=True)
        # self.data = self.data[self.data['virtual_traj_id'] == 0].reset_index(drop=True) # Center only
        '''
        self.data = self.data[self.data['is_train_candidate'] == 1].reset_index(drop=True)
        self.data = self.data.iloc[:450000, :]
        # self.img_dir = img_dir
        self.tokenizer = SimpleTokenizer(self.data['instruction'].astype(str).unique())
        self.encoded_instructions = [self.tokenizer.encode(str(i)) for i in self.data['instruction']]
        
        print(f"Loading {len(self.data)} images into RAM (uint8)...")
        self.images = {}
        for idx in tqdm(range(len(self.data))):
            img_name = self.data.iloc[idx]['image_path']
            # img_path = os.path.join(self.img_dir, img_name)
            img_path = os.path.join(img_name) 
            try:
                with Image.open(img_path) as img:
                    self.images[idx] = np.array(img, dtype=np.uint8)
            except:
                self.images[idx] = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        gc.collect()

class DiskCarlaDataset(Dataset):
    def __init__(self, dataframe, img_dir, tokenizer, indices, is_train=True):
        """
        Reads images from disk on-the-fly.
        """
        self.data = dataframe
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.indices = indices
        self.is_train = is_train
        
        # Pre-encode text for the specific indices we own
        # We create a lookup dict for speed
        self.encoded_instructions = {}
        for idx in self.indices:
            text = str(self.data.iloc[idx]['instruction'])
            self.encoded_instructions[idx] = self.tokenizer.encode(text)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        current_traj = self.data.iloc[idx]['trajectory_id']
        
        do_flip = self.is_train and (random.random() < 0.5)
        
        frames = []
        for t in range(STACK_FRAMES):
            target_idx = idx - t
            
            # Check continuity
            if target_idx < 0 or self.data.iloc[target_idx]['trajectory_id'] != current_traj:
                # Boundary: duplicate the last valid frame
                arr = frames[0] if frames else self._load_image(idx, do_flip)
                frames.insert(0, arr)
            else:
                frames.insert(0, self._load_image(target_idx, do_flip))

        # Stack numpy arrays: [4, H, W, 3] -> [12, H, W]
        img_stack = np.concatenate(frames, axis=2) 
        img_stack = np.transpose(img_stack, (2, 0, 1)) 
        
        text = self.encoded_instructions[idx]
        
        # --- TARGETS & SCALING ---
        row = self.data.iloc[idx]
        steer = row['steer']
        
        # 1. Flip Logic
        if do_flip: steer = -steer
        
        # 2. Amplification Logic
        steer = steer * CONTROL_SCALE

        throttle = row['throttle']
        brake = row['brake']
        speed = row['speed']
        accel = throttle - brake
        accel = accel * CONTROL_SCALE
        
        target_action = torch.tensor([steer, accel], dtype=torch.float32)
        
        # --- PREVIOUS ACTION ---
        if idx > 0 and self.data.iloc[idx-1]['trajectory_id'] == current_traj:
            prev_row = self.data.iloc[idx-1]
            p_steer = prev_row['steer']
            if do_flip: p_steer = -p_steer
            p_steer = p_steer * CONTROL_SCALE
            p_throttle = prev_row['throttle']
            p_brake = prev_row['brake']
            p_speed = prev_row['speed']
            p_accel = p_throttle - p_brake
            p_accel = p_accel * CONTROL_SCALE
            
            prev_action = torch.tensor([p_steer, p_accel, p_speed], dtype=torch.float32)
        else:
            prev_action = torch.zeros(3, dtype=torch.float32)
            
        return img_stack, text, prev_action, target_action

    def _load_image(self, idx, do_flip):
        img_name = self.data.iloc[idx]['image_path']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Use PIL for loading
        try:
            with Image.open(img_path) as img:
                arr = np.array(img, dtype=np.uint8)
        except:
            # Fallback black image
            arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            
        if do_flip:
            arr = np.fliplr(arr).copy()
            
        return arr

class CarlaFastDataset(Dataset):
    def __init__(self, cache_manager, indices, is_train=True):
        self.manager = cache_manager
        self.indices = indices
        self.is_train = is_train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        current_traj = self.manager.data.iloc[idx]['trajectory_id']
        do_flip = self.is_train and (random.random() < FLIP_PROB)
        
        frames = []
        for t in range(STACK_FRAMES):
            target_idx = idx - t
            if target_idx < 0 or self.manager.data.iloc[target_idx]['trajectory_id'] != current_traj:
                arr = frames[0] if frames else self._get_raw(idx, do_flip)
                frames.insert(0, arr)
            else:
                frames.insert(0, self._get_raw(target_idx, do_flip))

        img_stack = np.concatenate(frames, axis=2) 
        img_stack = np.transpose(img_stack, (2, 0, 1)) 

        # instruction = self.manager.encoded_instructions[idx]
        instruction = self.manager.data.iloc[idx]['instruction']
        if do_flip:
            if "left" in instruction:
                instruction = instruction.replace("left", "right")
            elif "right" in instruction:
                instruction = instruction.replace("right", "left")
        branch_id = command_to_branch_id(instruction)
        text = self.manager.tokenizer.encode(instruction)

        row = self.manager.data.iloc[idx]
        steer = row['steer']
        if do_flip: steer = -steer
        steer = steer * CONTROL_SCALE

        throttle = row['throttle']
        brake = row['brake']
        speed = row['speed']

        if speed > 2.0 and throttle < 0.01 and brake < 0.01:
            throttle = 0.2

        accel = throttle - brake
        accel = accel * CONTROL_SCALE
        target_action = torch.tensor([steer, accel], dtype=torch.float32)
        
        # Previous Action (Apply same cleaning)
        if idx > 0 and self.manager.data.iloc[idx-1]['trajectory_id'] == current_traj:
            prev_row = self.manager.data.iloc[idx-1]
            p_steer = prev_row['steer']
            if do_flip: p_steer = -p_steer
            p_steer = p_steer * CONTROL_SCALE

            p_throttle = prev_row['throttle']
            p_brake = prev_row['brake']
            p_speed = prev_row['speed']
            p_accel = p_throttle - p_brake
            p_accel = p_accel * CONTROL_SCALE
            
            prev_action = torch.tensor([p_steer, p_accel, p_speed], dtype=torch.float32)
        else:
            prev_action = torch.zeros(3, dtype=torch.float32)
            
        return img_stack, branch_id, text, prev_action, target_action

    def _get_raw(self, idx, do_flip):
        arr = self.manager.images[idx] 
        if do_flip: arr = np.fliplr(arr).copy()
        else: arr = arr.copy()
        return arr


def collate_fn(batch):
    imgs, branch_ids, texts, prevs, targets = zip(*batch)
    imgs = torch.from_numpy(np.stack(imgs)) 
    branch_ids = torch.tensor(branch_ids, dtype=torch.long)
    prevs = torch.stack(prevs)
    targets = torch.stack(targets)
    padded_texts = torch.zeros(len(texts), MAX_TOKEN_LEN, dtype=torch.long)
    for i, t in enumerate(texts): padded_texts[i, :len(t)] = t
    return imgs, padded_texts, branch_ids, prevs, targets

def compute_metrics(preds, targets):
    targets = targets[:, :2]  # Ignore speed for metrics
    abs_err = torch.abs(preds - targets)
    mae = torch.mean(abs_err).item()
    
    # Thresholds (Adjusted for Scale=5.0)
    steer_ok = abs_err[:, 0] < (0.05 * CONTROL_SCALE)
    accel_ok = abs_err[:, 1] < (0.05 * CONTROL_SCALE)
    
    correct = (steer_ok & accel_ok).float().sum().item()
    return mae, correct


def branched_loss_function(preds, targets, branch_ids):
    """
    preds: [Batch, 5, 2] (The outputs from all heads)
    targets: [Batch, 2] (True Steer, True Accel)
    branch_ids: [Batch] (0, 1, 2, 3, or 4)
    """
    batch_size = preds.shape[0]
    
    # 1. Select the specific branch prediction for each sample
    # We use torch.gather to pick the head corresponding to the command
    # indices shape needs to be [Batch, 1, 2] to gather steering and accel
    indices = branch_ids.view(batch_size, 1, 1).expand(batch_size, 1, 2)
    
    # relevant_preds: [Batch, 1, 2] -> [Batch, 2]
    relevant_preds = torch.gather(preds, 1, indices).squeeze(1)
    
    # 2. Now calculate loss normally using ONLY the relevant predictions
    error = torch.abs(relevant_preds - targets)
    
    # Steer Logic (Squared Weighting)
    steer_val = torch.abs(targets[:, 0])
    steer_mag_weight = 1.0 + 10.0 * (steer_val ** 2)
    
    steer_loss = (error[:, 0] * 25.0 * steer_mag_weight).mean()
    
    # Accel Logic
    is_active_accel = torch.abs(targets[:, 1]) > 0.05
    accel_weight = torch.ones_like(targets[:, 1]) * 2.0
    accel_weight[is_active_accel] = 10.0
    
    accel_loss = (error[:, 1] * accel_weight).mean()
    
    return steer_loss + accel_loss, [steer_loss.item(), accel_loss.item()]

def accel_robust_loss(preds, targets):
    # preds/targets: [Steer, Accel]
    
    # 1. Calculate raw L1 error
    error = torch.abs(preds - targets)
    
    # 2. Create a Magnitude-Aware weight for Steering
    steer_val = torch.abs(targets[:, 0])
    steer_magnitude_weight = 1.0 + 20.0 * steer_val
    steer_loss = (error[:, 0] * 15.0 * steer_magnitude_weight).mean()

    # steer_magnitude_weight = 1.0 + 10.0 * (steer_val ** 2)
    # steer_loss = (error[:, 0] * 25.0 * steer_magnitude_weight).mean()
    
    is_active_accel = torch.abs(targets[:, 1]) > 0.05
    accel_weight = torch.ones_like(targets[:, 1]) * 2.0
    accel_weight[is_active_accel] = 10.0   # 20.0

    accel_loss = (error[:, 1] * accel_weight).mean()
    
    total_loss = steer_loss + accel_loss
    
    return total_loss, [steer_loss.item(), accel_loss.item()]


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only')
    parser.add_argument('--model-path', type=str, required=False, help='Path to the model checkpoint for evaluation')
    parser.add_argument('--fast-dataset', action='store_true', help='Use fast in-RAM dataset (requires more RAM)')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--no-eval', action='store_true', help='Disable evaluation during training')
    args = parser.parse_args()

    eval_only = args.eval_only
    model_path = args.model_path
    fast_dataset = args.fast_dataset
    use_wandb = args.use_wandb
    if use_wandb:
        import wandb
        wandb.init(project="vla-accel", name="carla_vla_model_train")

    print(f"Initializing Training on {DEVICE}...")
    if not args.no_eval:
        client = carla.Client('localhost', CARLA_CLIENT_PORT)
        client.set_timeout(20.0)
        world = client.load_world('Town10HD_Opt')
        if EVAL_SCENARIO == "base":
            world.set_weather(carla.WeatherParameters.ClearNoon)
        elif EVAL_SCENARIO == "weather":
            weather = carla.WeatherParameters.HardRainNoon
            weather.fog_density = 10.0
            weather.wetness = 100.0
            world.set_weather(weather)
        elif EVAL_SCENARIO == "dynamic":
            world.set_weather(carla.WeatherParameters.ClearNoon)
            # traffic_actors = spawn_traffic(world, client, N_VEHICLES, N_WALKERS)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        world.apply_settings(settings)

    # 1. Load Metadata Only (Fast)
    def process_df(path, scenario_name, is_rainy, is_crowded):
        df = pd.read_csv(os.path.join(path, "data.csv"))
        # Prepend the folder path to the image filename
        df['image_path'] = os.path.join(path, "images") + "/" + df['image_path']
        # Add metadata tags for the Gating Network
        df['is_rainy'] = is_rainy
        df['is_crowded'] = is_crowded
        # Ensure trajectory_id is unique across the whole dataset
        # (Offset each by 1,000,000 to prevent overlap)
        offset = {"base": 0, "weather": 1000000, "dynamic": 2000000}
        df['trajectory_id'] += offset[scenario_name]
        return df

    df_base = process_df(BASE_DATASET_DIR, "base", 0, 0)
    df_weather = process_df(WEATHER_DATASET_DIR, "weather", 1, 0)
    df_dynamic = process_df(DYNAMIC_DATASET_DIR, "dynamic", 0, 1)
    full_df = pd.concat([df_base, df_weather, df_dynamic], ignore_index=True)
    # full_df = df_base

    '''
    def get_cam_code(path):
        if '_c.png' in path: return 0
        if '_l.png' in path: return 1
        if '_r.png' in path: return 2
        return 0
    full_df['cam_code'] = full_df['image_path'].apply(get_cam_code)
    full_df['virtual_traj_id'] = full_df['trajectory_id'] * 10 + full_df['cam_code']
    full_df = full_df.sort_values(['virtual_traj_id', 'image_path']).reset_index(drop=True)
    # full_df = full_df[full_df['virtual_traj_id'] == 0].reset_index(drop=True) # Center only
    '''
    full_df = full_df[full_df['is_train_candidate'] == 1].reset_index(drop=True)

    tokenizer = SimpleTokenizer(full_df['instruction'].astype(str).unique())
    model = CarlaVLA(STACK_FRAMES).to(DEVICE)

    if eval_only:
        model.eval()
        model.load_state_dict(torch.load(model_path), strict=True)
        completion, cte = evaluate_in_carla(model, client, 0, tokenizer, DEVICE, save_to_wandb=False)
        print(f"Completion: {completion:.2f}% | Avg CTE: {cte:.3f}m")
        return
 
    # 2. Filter Candidates
    valid_df = full_df
    valid_df = valid_df.iloc[:450000, :].reset_index(drop=True)
    
    train_indices = valid_df.index.tolist()
    
    print(f"Train Frames: {len(train_indices)}")
    
    if fast_dataset:
        print("Using Fast In-RAM Dataset...")
        cache_manager = CarlaImageCache(None, None)
        train_ds = CarlaFastDataset(cache_manager, train_indices, is_train=True)
    else:
        print("Using Disk-Based Dataset...")
        # train_ds = DiskCarlaDataset(valid_df, IMG_DIR, tokenizer, train_indices, is_train=True)
        exit(0)
    
    # 6. DataLoaders (High Workers for Disk I/O)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              prefetch_factor=PREFETCH_FACTOR, collate_fn=collate_fn)
    
    augmenter = GPUAugmenter().to(DEVICE) 
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = branched_loss_function
    scaler = GradScaler()
    
    best_val_mae = float('inf')
    best_val_loss = float('inf')
    noise_levels = torch.tensor([0.01, 0.01, 0.2]).to(DEVICE)

    best_completion = 0.0
    best_cte = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_losses = [0, 0]

        if epoch < 40:
            current_dropout = (epoch / 40.0) * ACTION_DROPOUT_PROB
        else:
            current_dropout = ACTION_DROPOUT_PROB

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, texts, branch_ids, prevs, targets in loop:
            imgs = imgs.to(DEVICE, non_blocking=True)
            texts = texts.to(DEVICE, non_blocking=True)
            branch_ids = branch_ids.to(DEVICE, non_blocking=True)
            prevs = prevs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            
            imgs = augmenter(imgs, is_train=True)
            
            if random.random() < current_dropout:
                prevs = torch.zeros_like(prevs)
            else:
                noise = torch.randn_like(prevs) * noise_levels
                prevs = prevs + noise
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                preds = model(imgs, prevs)
                # loss, all_losses = criterion(preds, targets, [str(t) for t in texts])
                # loss, all_losses = criterion(preds, targets)
                loss, all_losses = criterion(preds, targets, branch_ids)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            for i in range(len(total_losses)):
                total_losses[i] += all_losses[i]
            loop.set_postfix(loss=loss.item(), steer_loss=all_losses[0], accel_loss=all_losses[1])
            
        model.eval()
      
        steer_loss = total_losses[0] / len(train_loader)
        accel_loss = total_losses[1] / len(train_loader)
        print_str = f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.5f} | Steer Loss: {steer_loss:.5f} | Accel Loss: {accel_loss:.5f}"
        if not args.no_eval:
            completion, cte = evaluate_in_carla(model, client, epoch, tokenizer, DEVICE, save_to_wandb=use_wandb)
            print_str += f" | Completion: {completion:.2f}% | Avg CTE: {cte:.3f}m"

            if completion > 50.0:
                if cte < best_cte:
                    best_cte = cte
                    torch.save(model.state_dict(), "carla_vla_best_driver.pth")
                    print(f">>> NEW BEST DRIVER (CTE {cte:.3f}m)")
            
            if completion > best_completion: # Initialize best_completion = 0.0
                best_completion = completion
                torch.save(model.state_dict(), "carla_vla_furthest_driver.pth")
                print(f">>> NEW FURTHEST DRIVER ({completion:.1f}%)")

        print(print_str)

        if use_wandb:
            wandb.log({
                "Train Loss": total_loss/len(train_loader),
                "Steer Loss": total_losses[0]/len(train_loader),
                "Accel Loss": total_losses[1]/len(train_loader),
                "Epoch": epoch
            }, step=epoch)
            if not args.no_eval:
                wandb.log({
                    "Completion": completion,
                    "Avg CTE": cte
                }, step=epoch)
            torch.save(model.state_dict(), f"carla_vla_epoch_{epoch}.pth")
            wandb.save(f"carla_vla_epoch_{epoch}.pth")
            

 
if __name__ == "__main__":
    train()
