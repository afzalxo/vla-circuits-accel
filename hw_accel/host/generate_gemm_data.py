import torch
import torch.nn.functional as F
import numpy as np
import os

# --- CONFIGURATION ---
OUTPUT_DIR = "fpga_data_gemm"
M = 1  # Batch Size / Sequence Length (Mapped to Width)
K = 4096  # Input Features (Mapped to In Channels)
N = 512  # Output Features (Mapped to Out Channels)

# Hardware Parameters
IC_PAR = 16
OC_PAR = 16


torch.manual_seed(3233)

def save_binary(data, filename):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    path = os.path.join(OUTPUT_DIR, filename)
    # Ensure data is flat and int8
    data.flatten().astype(np.int8).tofile(path)
    print(f"Saved {path}: Shape {data.shape}, Dtype {data.dtype}")

def generate_gemm():
    print(f"--- Generating GEMM Data (M={M}, K={K}, N={N}) ---")
    
    # 1. Generate Random Input (X)
    # Shape for Conv2d: [Batch, Channels, Height, Width]
    # We map GEMM M -> Width, K -> Channels. Height = 1.
    # Range: -64 to 64 to avoid massive overflows before shift
    input_t = torch.randint(-64, 64, (1, K, 1, M)).float()
    
    # 2. Generate Random Weights (W)
    # Shape for Conv2d: [Out_Channels, In_Channels, kH, kW]
    # 1x1 Kernel
    weights_t = torch.randint(-64, 64, (N, K, 1, 1)).float()
    
    # 3. Perform GEMM (via 1x1 Conv)
    # Result is Int32 (Accumulator)
    acc_int = F.conv2d(input_t, weights_t, stride=1, padding=0)
    
    # 4. Calculate Optimal Quantization Shift
    # We want the max value to fit into int8 range [-128, 127]
    max_val = torch.max(torch.abs(acc_int)).item()
    
    if max_val > 0:
        # log2(max / 127)
        needed_shift = np.ceil(np.log2(max_val / 127.0))
        quant_shift = int(max(0, needed_shift))
    else:
        quant_shift = 0
        
    print(f"Max Accumulator: {max_val}")
    print(f"Recommended Quant Shift: {quant_shift}")
    
    # 5. Quantize (Shift & Clamp)
    # Use floor_divide to match Verilog >>>
    output_scaled = torch.floor_divide(acc_int, (2 ** quant_shift))
    output_int8 = torch.clamp(output_scaled, -128, 127).char() # int8
    
    # 6. Save Files
    
    # Input: Save as Planar [C, H, W] -> [K, 1, M]
    # Your C++ pack_feature_map expects this layout.
    # print("Feature map first 10 rows:")
    # print(input_t[0, :, 0, 0:10].transpose(1, 0))
    # print("Weights first 10 columns:")
    # print(weights_t[0:17, :, 0, 0])
    save_binary(input_t.byte().numpy(), "input_layer0.bin")
    
    # Weights: Save as [OC, IC, 1, 1] -> [N, K]
    # Your C++ pack_weights (with is_1x1=True) expects this layout.
    save_binary(weights_t.byte().numpy(), "weights_layer0.bin")
    
    # Output: Save as Planar [OC, H, W] -> [N, 1, M]
    save_binary(output_int8.numpy(), "golden_output_layer0.bin")
    
    print("\n--- Host C++ Configuration ---")
    print(f"instr.width       = {M};")
    print(f"instr.height      = 1;")
    print(f"instr.in_channels = {K};")
    print(f"instr.out_channels= {N};")
    print(f"instr.quant_shift = {quant_shift};")
    print(f"instr.opcode      = OP_GEMM;")
    print(f"instr.is_sparse   = 0;")

if __name__ == "__main__":
    generate_gemm()
