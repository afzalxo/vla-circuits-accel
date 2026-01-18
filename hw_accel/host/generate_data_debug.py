import torch
import torch.nn.functional as F
import numpy as np
import os

# --- CONFIGURATION ---
OUTPUT_DIR = "fpga_data_debug"
torch.manual_seed(10)

H = 8
W = 8
IC = 64
OC = 64
OC_DENSE = 512
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
RELU_EN = True

# Hardware Constants
IC_PAR = 16
OC_PAR = 16
PP_PAR = 8
TILE_HEIGHT = 4
HBM_DATA_WIDTH = 512 # bits
DATA_WIDTH = 8 # bits

def save_binary(data, filename):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    path = os.path.join(OUTPUT_DIR, filename)
    data.tofile(path)
    print(f"Saved {filename}: Shape {data.shape}, Dtype {data.dtype}, Bytes {data.nbytes}")

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
    IC_PAR = 16
    PP_PAR = 8
    TILE_HEIGHT = 4
    
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


def generate_data():
    print(f"--- Generating Flatten Debug Data (H={H}, W={W}, IC={IC}, OC={OC}) ---")
    
    # 1. Generate Random Input
    # Shape: [1, IC, H, W]
    input_t = torch.randint(-32, 32, (1, IC, H, W)).float()
    
    # 2. Generate Random Weights
    # Shape: [OC, IC, 3, 3]
    weights_t = torch.randint(-32, 32, (OC, IC, KERNEL_SIZE, KERNEL_SIZE)).float()
    weights_dense = torch.randint(-32, 32, (OC_DENSE, OC * 8 * 8)).float()
    weights_dense1 = torch.randint(-32, 32, (OC_DENSE // 2, OC_DENSE)).float()
    
    output_int8, quant_shift = simulate_hardware_conv(input_t, weights_t, STRIDE, PADDING, RELU_EN)
    
    print(f"Conv Layer Quantization Shift: {quant_shift}")
    # 5. Save Inputs/Weights for Host
    save_binary(input_t.byte().numpy(), "input_layer_debug.bin")
    save_binary(weights_t.byte().numpy(), "weights_layer_debug.bin")
    save_binary(output_int8.byte().numpy(), "golden_output_conv.bin")
    
    output_int8 = output_int8.flatten(1)
    output_dense_int8, quant_shift_dense = simulate_hardware_dense(output_int8, weights_dense, RELU_EN)
    save_binary(output_dense_int8.byte().numpy(), "golden_output_dense.bin")
    weights_dense_reordered = reorder_weights_for_hardware(weights_dense.view(OC_DENSE, OC, 8, 8), OC, 8, 8)
    save_binary(weights_dense_reordered.byte().numpy(), "weights_dense_debug.bin")
    print(f"Dense Layer Quantization Shift: {quant_shift_dense}")
	
    out_dense1_int8, quant_shift_dense1 = simulate_hardware_dense(output_dense_int8, weights_dense1, RELU_EN)
    save_binary(out_dense1_int8.byte().numpy(), "golden_output_dense1.bin")
    save_binary(weights_dense1.byte().numpy(), "weights_dense1_debug.bin")
    print(f"Dense1 Layer Quantization Shift: {quant_shift_dense1}")
    # --- GENERATE MEMORY IMAGES ---
    
    print("\n--- Host Configuration ---")
    print(f"instr.width       = {W};")
    print(f"instr.height      = {H};")
    print(f"instr.in_channels = {IC};")
    print(f"instr.out_channels= {OC};")
    print(f"instr.stride      = {STRIDE};")
    print(f"instr.quant_shift = {quant_shift};")
    print(f"instr.relu_en     = {1 if RELU_EN else 0};")
    print(f"instr.log2_mem_tile_height = 2; // log2(4)")

if __name__ == "__main__":
    generate_data()
