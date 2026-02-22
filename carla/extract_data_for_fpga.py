import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import cv2
import json
from tqdm import tqdm
from models.carla_model import CarlaVLAHW as CarlaVLA

# --- CONFIGURATION ---
DATASET_DIR = "carla_dataset_1m"
CSV_PATH = os.path.join(DATASET_DIR, "data.csv")
IMG_DIR = os.path.join(DATASET_DIR, "images")
MODEL_PATH = "model_hw.pth"
OUTPUT_DIR = "fpga_data_global"

# Hardware Constraints
BIT_WIDTH = 8
Q_MAX = 127
Q_MIN = -128
STACK_FRAMES = 1
IMG_SIZE = 256
CALIBRATION_SAMPLES = 1000  # Number of samples to profile
ROUNDING_MODE = "round"

# Preprocessing Constants
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_input_data(df, sample_idx):
    """Prepares 4-frame stack and prev_action exactly like the Eval script."""
    row = df.iloc[sample_idx]
    current_traj = row['trajectory_id']
    
    # 1. Image Stacking
    frames = []
    for t in range(STACK_FRAMES):
        target_idx = sample_idx - t
        if target_idx < 0 or df.iloc[target_idx]['trajectory_id'] != current_traj:
            target_idx = sample_idx
        
        img_path = os.path.join(IMG_DIR, df.iloc[target_idx]['image_path'])
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(img_res).permute(2, 0, 1).float() / 255.0
        frames.insert(0, tensor)
    
    input_stack = torch.cat(frames, dim=0).unsqueeze(0)
    input_stack = (input_stack.reshape(-1, 3, 256, 256) - NORM_MEAN) / NORM_STD
    input_stack = input_stack.reshape(1, 3 * STACK_FRAMES, 256, 256)

    # 2. Prev Action
    if sample_idx > 0 and df.iloc[sample_idx-1]['trajectory_id'] == current_traj:
        p_row = df.iloc[sample_idx-1]
        # prev_action = torch.tensor([p_row['steer'], (p_row['throttle'] - p_row['brake']), p_row['speed']])
        prev_action = torch.tensor([
            p_row['steer'], 
            (p_row['throttle'] - p_row['brake']), 
            p_row['speed']
        ], dtype=torch.float32)
    else:
        prev_action = torch.zeros(3)

    # prev_action = torch.zeros(3)
    
    return input_stack, prev_action.unsqueeze(0)

def save_binary(data, filename):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    path = os.path.join(OUTPUT_DIR, filename)
    data.to(torch.int8).numpy().tofile(path)

def hardware_round_shift(acc, shift):
    """Bit-accurate rounding: (acc + bias) >> shift"""
    if ROUNDING_MODE == "round":
        if shift > 0:
            bias = 1 << (shift - 1)
            return torch.floor_divide(acc + bias, 2**shift)
    elif ROUNDING_MODE == "floor":
        if shift > 0:
            return torch.floor_divide(acc, 2**shift)
    return acc * (2**abs(shift))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_idx', type=int, default=224, help="Sample to export as golden test vector")
    args = parser.parse_args()

    print("Loading Model and Data...")
    df = pd.read_csv(CSV_PATH)
    model = CarlaVLA(STACK_FRAMES)
    sd = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict({k: v for k, v in sd.items() if 'bias' not in k}, strict=True)
    model.eval()

    # =========================================================================
    # PHASE 1: GLOBAL PROFILING
    # =========================================================================
    print(f"--- Phase 1: Profiling Activations ({CALIBRATION_SAMPLES} samples) ---")
    calib_indices = df.sample(n=CALIBRATION_SAMPLES, random_state=42).index.tolist()
    
    # We track the max absolute value seen at every activation point
    act_maxima = {}

    with torch.no_grad():
        for idx in tqdm(calib_indices):
            img_stack, prev_act = get_input_data(df, idx)
            
            # Step-by-step forward pass to capture activation ranges
            curr = img_stack
            for i, layer in enumerate(model.get_conv_layers()):
                stride = 2 if i < 5 else 1
                curr = F.relu(F.conv2d(curr, layer.weight, bias=None, stride=stride, padding=1))
                act_maxima[f'conv{i}'] = max(act_maxima.get(f'conv{i}', 0), torch.max(torch.abs(curr)).item())

            curr = F.adaptive_avg_pool2d(curr, (1, 1)).flatten(1)
            act_maxima['gap'] = max(act_maxima.get('gap', 0), torch.max(torch.abs(curr)).item())

            # prev_act = torch.tensor(prev_act, dtype=torch.double)
            # print(prev_act)
            act_feat = F.relu(model.measure_proj(prev_act))
            act_maxima['measure'] = max(act_maxima.get('measure', 0), torch.max(torch.abs(act_feat)).item())

            curr = torch.cat([curr, act_feat], dim=1)
            
            dense_layers = [("fusion", model.fusion, True), ("fc1", model.branches[0][0], True), 
                            ("fc2", model.branches[0][2], True), ("head", model.branches[0][4], False)]
            
            for name, layer, use_relu in dense_layers:
                curr = F.linear(curr, layer.weight, bias=None)
                if use_relu: curr = F.relu(curr)
                act_maxima[name] = max(act_maxima.get(name, 0), torch.max(torch.abs(curr)).item())

    # =========================================================================
    # PHASE 2: CALCULATE FIXED GLOBAL SHIFTS
    # =========================================================================
    print("--- Phase 2: Freezing Global Shifts ---")
    global_shifts = {}
    
    # Track cumulative scale throughout profiling logic
    # We use a representative input scale (99.9th percentile of input)
    input_stack_example, _ = get_input_data(df, args.sample_idx)
    c_scale = Q_MAX / torch.max(torch.abs(input_stack_example)).item()
    
    for i in range(len(model.get_conv_layers())):
        w_max = torch.max(torch.abs(model.get_conv_layers()[i].weight)).item()
        w_scale = Q_MAX / w_max
        c_scale *= w_scale
        
        # Shift = ceil(log2( (Float_Max * Cumulative_Scale) / Q_MAX ))
        val_to_quantize = act_maxima[f'conv{i}'] * c_scale
        shift = int(np.ceil(np.log2(val_to_quantize / Q_MAX))) if val_to_quantize > 0 else 0
        global_shifts[f'conv{i}'] = shift
        c_scale /= (2**shift)

    # GAP
    h, w = 8, 8 # Conv Output dims
    gap_shift = int(np.log2(h * w))
    global_shifts['gap'] = gap_shift
    c_scale *= ((h*w) / (2**gap_shift))

    # Dense
    for name, layer, _ in dense_layers:
        w_scale = Q_MAX / torch.max(torch.abs(layer.weight)).item()
        c_scale *= w_scale
        val_to_quantize = act_maxima[name] * c_scale
        shift = int(np.ceil(np.log2(val_to_quantize / Q_MAX))) if val_to_quantize > 0 else 0
        global_shifts[name] = shift
        c_scale /= (2**shift)

    with open(os.path.join(OUTPUT_DIR, "global_shifts.json"), "w") as f:
        json.dump(global_shifts, f, indent=4)

    # =========================================================================
    # PHASE 3: EXTRACTION (SPECIFIC SAMPLE)
    # =========================================================================
    print(f"--- Phase 3: Extracting Golden Vectors for Sample {args.sample_idx} ---")
    img_stack, prev_act = get_input_data(df, args.sample_idx)
    # [1, C, H, W]
    
    # 1. Input Vision
    in_max = torch.max(torch.abs(img_stack)).item()
    print(in_max)
    in_scale = Q_MAX / in_max
    print(in_scale)
    curr_int = (img_stack * in_scale).round().clamp(Q_MIN, Q_MAX)
    print(curr_int[0, 0, :15, :15])
    save_binary(curr_int, "input_vision.bin")
    
    # 2. Conv Layers
    cumulative_scale = in_scale
    conv_layers = model.get_conv_layers()
    for i, layer in enumerate(conv_layers):
        w_max = torch.max(torch.abs(layer.weight)).item()
        w_scale = Q_MAX / w_max
        w_int = (layer.weight * w_scale).round().clamp(Q_MIN, Q_MAX)
        save_binary(w_int, f"weights_conv{i}.bin")
        
        stride = 2 if i < 5 else 1
        acc = F.conv2d(curr_int.float(), w_int.float(), bias=None, stride=stride, padding=1)
        acc = F.relu(acc)
        
        shift = global_shifts[f'conv{i}']
        curr_int = hardware_round_shift(acc, shift).clamp(Q_MIN, Q_MAX)
        save_binary(curr_int, f"golden_conv{i}.bin")
        cumulative_scale = (cumulative_scale * w_scale) / (2**shift)

    # 3. GAP
    gap_acc = torch.sum(curr_int.float(), dim=(2, 3), keepdim=True)
    curr_int = hardware_round_shift(gap_acc, global_shifts['gap']).clamp(Q_MIN, Q_MAX).flatten(1)
    save_binary(curr_int, "golden_gap.bin")
    cumulative_scale *= (8*8 / (2**global_shifts['gap']))

    # 4. Fusion
    prev_act = torch.zeros_like(prev_act)
    act_feat_float = F.relu(model.measure_proj(prev_act))
    act_feat_int = (act_feat_float * cumulative_scale).round().clamp(Q_MIN, Q_MAX)
    print(cumulative_scale)
    # print(act_feat_int)
    save_binary(act_feat_int, "extra_features.bin")
    
    curr_int = torch.cat([curr_int, act_feat_int], dim=1)
    save_binary(curr_int, "golden_fused_features.bin")

    # 5. Dense
    for name, layer, use_relu in dense_layers:
        w_max = torch.max(torch.abs(layer.weight)).item()
        w_scale = Q_MAX / w_max
        w_int = (layer.weight * w_scale).round().clamp(Q_MIN, Q_MAX)
        save_binary(w_int, f"weights_{name}.bin")
        
        acc = F.linear(curr_int.float(), w_int.float(), bias=None)
        if use_relu: acc = F.relu(acc)
        
        shift = global_shifts[name]
        curr_int = hardware_round_shift(acc, shift).clamp(Q_MIN, Q_MAX)
        save_binary(curr_int, f"golden_{name}.bin")
        cumulative_scale = (cumulative_scale * w_scale) / (2**shift)

    # =========================================================================
    # FINAL VERIFICATION
    # =========================================================================
    hw_logits = curr_int.float() / cumulative_scale
    hw_logits = hw_logits.detach()
    hw_action = torch.tanh(hw_logits).numpy()[0]
    
    with torch.no_grad():
        sw_logits = model(img_stack, prev_act)[0, 0] # Branch 0
        sw_action = torch.tanh(sw_logits).numpy()

    print("\n" + "="*60)
    print(f"GLOBAL STATIC QUANTIZATION RESULTS")
    print("-" * 60)
    print(f"HW Logits (No Scaling): {curr_int}")
    print(f"HW Action (Fixed Shifts): {hw_action}")
    print(f"SW Action (Ideal Float):  {sw_action}")
    print(f"Final Global Scale:       {cumulative_scale:.4f}")
    
    err = np.abs(hw_action - sw_action)
    print(f"Max Deviation:            {np.max(err):.4f}")
    if np.max(err) < 0.1:
        print("\nSUCCESS: Global shifts provide stable, accurate results.")
    else:
        print("\nWARNING: High error. Consider increasing calibration samples.")
    print("="*60)

if __name__ == "__main__":
    main()
