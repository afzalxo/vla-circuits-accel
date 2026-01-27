import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

# Import your model definition
from models.carla_model import CarlaVLA, SimpleTokenizer

# --- CONFIGURATION ---
DATASET_DIR = "carla_vla_dataset_hd" 
CSV_PATH = os.path.join(DATASET_DIR, "data.csv")
IMG_DIR = os.path.join(DATASET_DIR, "images")
MODEL_PATH = "carla_vla_lasso.pth"
OUTPUT_DIR = "fpga_data_carla"

torch.manual_seed(42)
DEVICE = torch.device("cpu") 
STACK_FRAMES = 4
IMG_SIZE = 128
INPUT_SCALE = 127.0 
IC_PAR = 32
PP_PAR = 2
TILE_HEIGHT = 4

def save_binary(data, filename):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    path = os.path.join(OUTPUT_DIR, filename)
    data.tofile(path)
    print(f"Saved {filename}: Shape {data.shape}, Dtype {data.dtype}")

def quantize_tensor(tensor):
    """Scales tensor to Int8 range (-128, 127) based on max abs value."""
    t_max = torch.max(torch.abs(tensor)).item()
    if t_max == 0: return tensor.byte(), 1.0
    scale = 127.0 / t_max
    t_quant = (tensor * scale).round().char() # int8
    return t_quant, scale

def simulate_hardware_conv(input_int8, weights_int8, stride, padding, relu=True):
    """Simulates FPGA Conv Layer"""
    input_f = input_int8.float()
    weights_f = weights_int8.float()
    
    acc_int = F.conv2d(input_f, weights_f, bias=None, stride=stride, padding=padding)
    
    if relu:
        acc_int = torch.relu(acc_int)
        
    max_acc = torch.max(torch.abs(acc_int)).item()
    if max_acc > 0:
        needed_shift = np.ceil(np.log2(max_acc / 127.0))
        quant_shift = int(max(0, needed_shift))
    else:
        quant_shift = 0
        
    output_scaled = torch.floor_divide(acc_int, (2 ** quant_shift))
    output_int8 = torch.clamp(output_scaled, -128, 127).char()
    
    return output_int8, quant_shift

def simulate_hardware_dense(input_int8, weights_int8, relu=True):
    """Simulates FPGA GEMM (Dense Layer)"""
    input_f = input_int8.float()
    weights_f = weights_int8.float()
    
    # Linear: y = xA^T
    acc_int = F.linear(input_f, weights_f)
    
    if relu:
        acc_int = torch.relu(acc_int)
        
    max_acc = torch.max(torch.abs(acc_int)).item()
    if max_acc > 0:
        needed_shift = np.ceil(np.log2(max_acc / 127.0))
        quant_shift = int(max(0, needed_shift))
    else:
        quant_shift = 0
        
    output_scaled = torch.floor_divide(acc_int, (2 ** quant_shift))
    output_int8 = torch.clamp(output_scaled, -128, 127).char()
    
    return output_int8, quant_shift

def reorder_weights_for_hardware(weight_tensor_chw, C, H, W):
    """
    Reorders weights to match the FPGA's Flatten/Pad Output Mode.
    
    Hardware Write Order (Blocked Planar):
    1. Tile Y       (0 .. H/4)
    2. Tile OC      (0 .. C/16)
    3. Row          (0 .. 4)
    4. Strip        (0 .. W/8)
    5. Pixel        (0 .. 8)
    
    The Dense Layer reads this stream. For each valid pixel written by the Conv layer,
    the Dense layer performs one MAC operation block.
    We must provide the weights in the exact same order.
    
    Args:
        weight_tensor_chw: [Out, C, H, W] (Standard PyTorch Shape)
    """
    Out_Dim = weight_tensor_chw.shape[0]
    
    num_h_tiles = (H + TILE_HEIGHT - 1) // TILE_HEIGHT
    num_c_tiles = (C + IC_PAR - 1) // IC_PAR
    num_w_strips = (W + PP_PAR - 1) // PP_PAR
    
    new_weights_list = []
    
    # Iterate exactly as the Hardware Output DMA iterates
    for ty in range(num_h_tiles):
        rows_remaining = H - (ty * TILE_HEIGHT)
        active_h = min(rows_remaining, TILE_HEIGHT)
        
        for toc in range(num_c_tiles):
            for r in range(active_h):
                global_h = (ty * TILE_HEIGHT) + r
                
                for ws in range(num_w_strips):
                    for p in range(PP_PAR):
                        global_w = (ws * PP_PAR) + p
                        
                        # Only extract weights for valid pixels.
                        # The hardware writes padding for invalid pixels (boundary),
                        # but we handle that by ensuring the weight is 0.
                        
                        if global_w < W:
                            # Extract 16 channels for this pixel
                            # Slice: [Out, C_start:C_end, H, W]
                            c_start = toc * IC_PAR
                            c_end = c_start + IC_PAR
                            
                            # Shape: [Out, 16]
                            block = weight_tensor_chw[:, c_start:c_end, global_h, global_w]
                            new_weights_list.append(block)
                        else:
                            # Image Boundary Padding
                            # The hardware writes dummy data here. We provide 0 weights.
                            zero_block = torch.zeros((Out_Dim, IC_PAR), dtype=weight_tensor_chw.dtype)
                            new_weights_list.append(zero_block)
                            
                        # NOTE: We do NOT append a "Padding Block" here.
                        # The hardware padding (Lanes 1-7) is handled by the Compute Unit
                        # using the SAME weight block we just appended.

    # Concatenate along the input dimension
    return torch.cat(new_weights_list, dim=1)



def extract():
    # --- 1. LOAD RESOURCES ---
    print("Loading Model...")
    df = pd.read_csv(CSV_PATH)
    tokenizer = SimpleTokenizer(df['instruction'].unique())
    
    model = CarlaVLA(tokenizer.vocab_size).to(DEVICE)
    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    new_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=True)
    model.eval()

    # --- 2. PREPARE INPUT ---
    SAMPLE_IDX = 100 
    row = df.iloc[SAMPLE_IDX]
    print(f"Extracting sample {SAMPLE_IDX}")
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frames = []
    for i in range(STACK_FRAMES):
        target_idx = max(0, SAMPLE_IDX - i)
        img_path = os.path.join(IMG_DIR, df.iloc[target_idx]['image_path'])
        with Image.open(img_path) as img:
            frames.insert(0, transform(img))
    
    input_tensor = torch.cat(frames, dim=0).unsqueeze(0) # [1, 12, 128, 128]

    # --- 3. QUANTIZE INPUT ---
    input_int8 = (input_tensor * INPUT_SCALE).round().char()
    save_binary(input_int8.numpy(), "input_vision.bin")
    
    # =========================================================================
    # CONVOLUTION LAYERS (1-5)
    # =========================================================================
    print("\n--- Processing Convolution Layers ---")
    
    current_input = input_int8
    conv_layers = model.get_conv_layers() 
    
    for i, layer in enumerate(conv_layers):
        print(f"Layer {i} (Conv): In={current_input.shape}")
        
        # 1. Weights
        w = layer.weight.detach()
        w_int8, scale = quantize_tensor(w)
        save_binary(w_int8.numpy(), f"weights_conv{i}.bin")
        
        current_stride = 1 if i == 4 else 2

        # 2. Simulate
        out_int8, shift = simulate_hardware_conv(
            current_input, w_int8, stride=current_stride, padding=1, relu=True
        )
        
        save_binary(out_int8.numpy(), f"golden_output_conv{i}.bin")
        print(f"  Shift: {shift}, Out={out_int8.shape}")
        
        current_input = out_int8

    # =========================================================================
    # DENSE LAYERS (VISION ONLY)
    # =========================================================================
    print("\n--- Processing Dense Layers (Vision Only) ---")
    
    # 1. PREPARE INPUT
    # Current 'current_input' is Conv5 Output: [1, 256, 4, 4] (Planar)
    # Hardware stores this as HWC [4, 4, 256]
    # We flatten it in HWC order to simulate the hardware reading it linearly
    # vis_hwc = current_input.permute(0, 2, 3, 1).flatten(1) # [1, 4096]
    vis_hwc = current_input.flatten(1)
    
    # 2. FUSION FC1 (SLICED)
    print("Layer FC1 (Vision Slice)")
    w_fc1 = model.fusion_fc1.weight.detach() # [512, 4224]
    
    # SLICE: Keep only the first 4096 columns (Vision)
    w_fc1_vis = w_fc1[:, :4096]
    
    print("  Reordering weights to Blocked Planar...")
    # w_fc1_reordered = reorder_weights_for_hardware(w_fc1_vis.view(512, 64, 8, 8), C=64, H=8, W=8)
    w_fc1_reordered = reorder_weights_for_hardware(w_fc1_vis.view(512, 256, 4, 4), C=256, H=4, W=4)
    
    w_fc1_int8, _ = quantize_tensor(w_fc1_reordered)
    save_binary(w_fc1_int8.numpy(), "weights_fc1.bin")

    # Simulate GEMM
    w_fc1_std_int8, _ = quantize_tensor(w_fc1_vis)
    out_fc1, shift_fc1 = simulate_hardware_dense(vis_hwc, w_fc1_std_int8, relu=True)
    save_binary(out_fc1.numpy(), "golden_output_fc1.bin")
    print(f"  Shift: {shift_fc1}, Out={out_fc1.shape}")
    
    # 3. FUSION FC2
    print("Layer FC2")
    w_fc2 = model.fusion_fc2.weight.detach()
    w_fc2_int8, _ = quantize_tensor(w_fc2)
    save_binary(w_fc2_int8.numpy(), "weights_fc2.bin")
    
    out_fc2, shift_fc2 = simulate_hardware_dense(out_fc1, w_fc2_int8, relu=True)
    save_binary(out_fc2.numpy(), "golden_output_fc2.bin")
    print(f"  Shift: {shift_fc2}, Out={out_fc2.shape}")
    
    # 4. ACTION HEAD
    print("Layer Action Head")
    w_head = model.action_head.weight.detach()
    w_head_int8, _ = quantize_tensor(w_head)
    save_binary(w_head_int8.numpy(), "weights_head.bin")
    
    out_head, shift_head = simulate_hardware_dense(out_fc2, w_head_int8, relu=False)
    save_binary(out_head.numpy(), "golden_output_head.bin")
    print(f"  Shift: {shift_head}, Out={out_head.shape}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n--- Hardware Configuration Summary ---")
    print("Conv Layers (All Stride 2, ReLU):")
    print("  L0: In=12(Pad16), Out=64")
    print("  L1: In=64, Out=64")
    print("  L2: In=64, Out=64")
    print("  L3: In=64, Out=64")
    print("  L4: In=64, Out=64")
    print("Dense Layers (Vision Only):")
    print("  FC1: In=4096, Out=512, ReLU")
    print("  FC2: In=512, Out=256, ReLU")
    print("  Head: In=256, Out=3(Pad16), No ReLU")

if __name__ == "__main__":
    extract()
