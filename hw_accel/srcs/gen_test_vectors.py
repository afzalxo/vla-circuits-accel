import torch
import torch.nn.functional as F
import numpy as np

# CONFIG
IC = 16
OC = 16
H = 64
W = 64 # Must be multiple of 8
PP = 8 # Pixel Parallelism
STRIPS = W // PP

def int_to_hex(val, bits):
    """Converts integer to hex string with specific bit width, handling 2's complement"""
    val = int(val)
    if val < 0:
        val = (1 << bits) + val
    return f"{val:0{bits//4}X}"

def pack_strip(strip_tensor):
    """
    Packs a [8, 16] tensor (8 pixels, 16 channels) into a single hex line.
    Layout: Pixel 7 (Ch15..0) ... Pixel 0 (Ch15..0)
    """
    # strip_tensor shape: [8, 16]
    hex_str = ""
    # Iterate pixels high to low (Verilog bit order)
    for p in range(PP-1, -1, -1):
        for c in range(IC-1, -1, -1):
            val = strip_tensor[p, c].item()
            hex_str += int_to_hex(val, 8)
    return hex_str

def pack_weights(weight_tensor):
    """
    Packs [16, 16] weights into a single hex line.
    Layout: Out15(In15..0) ... Out0(In15..0)
    """
    hex_str = ""
    for o in range(OC-1, -1, -1):
        for i in range(IC-1, -1, -1):
            val = weight_tensor[o, i].item()
            hex_str += int_to_hex(val, 8)
    return hex_str

def main():
    print(f"Generating Test Vectors for Image {H}x{W}...")

    # 1. Generate Random Data (Int8 range)
    # Inputs: [1, IC, H, W]
    img = torch.randint(-10, 10, (1, IC, H, W)).float()
    # print(img[0, 0, :, :])
    # img = torch.ones_like(img)
    '''
    img[0, :, 0, :W//2] = 1.0
    img[0, :, 0, W//2:] = 2.0
    img[0, :, 1, :W//2] = 3.0
    img[0, :, 1, W//2:] = 4.0
    img[0, :, 2, :W//2] = 5.0
    img[0, :, 2, W//2:] = 6.0
    '''
    
    # Weights: [OC, IC, 3, 3]
    # For this specific accelerator test, we assume the weights are SHARED 
    # across the 3x3 kernel (spatial invariance) or the controller handles loading.
    # *Correction*: The `vector_compute_unit` takes a [OC][IC] weight matrix.
    # The `conv_controller` loops 9 times. 
    # In a real DPU, weights change every cycle of the 9-cycle loop.
    # In our simplified `tb_integration`, we kept weights constant.
    # To match the Verilog exactly, let's assume weights are constant 1 for now, 
    # OR we assume the Verilog inputs a single weight matrix that is reused.
    # Let's generate random weights [OC, IC] and assume they are used for all 3x3 positions
    # (effectively a 1x1 conv smeared over 3x3, or a sum-pool).
    weights_core = torch.randint(0, 3, (OC, IC)).float()
    # print(weights_core[0, :])
    # weights_core = torch.ones_like(weights_core)
    
    # 2. Compute Golden Output (PyTorch)
    # We simulate the hardware behavior:
    # The hardware accumulates 9 positions. 
    # Since we feed the SAME weights_core for all 9 positions in the Verilog (it's a static input port),
    # Mathematically this is: Output = Input * Weight * 9 (if input is constant)
    # But input slides. So it is: Output = Conv2d(Input, Weights_Expanded)
    
    # Expand weights to 3x3: [OC, IC, 3, 3]
    # Each spatial position has the SAME weight matrix
    weights_3x3 = weights_core.unsqueeze(2).unsqueeze(3).repeat(1, 1, 3, 3)
    
    # Run Conv2d (Padding=1 to match hardware "Same" padding logic)
    # Note: Our hardware sequencer pads with 0 at boundaries.
    out_golden = F.conv2d(img, weights_3x3, padding=1).int()
    
    # 3. Dump Input Image Hex
    # Format: Line 0 = Row 0 Strip 0, Line 1 = Row 0 Strip 1...
    # Input tensor is [1, 16, 16, 16] -> Permute to [H, W, C] for easier slicing
    img_perm = img.squeeze(0).permute(1, 2, 0).int() # [H, W, C]
    
    with open("image_input.hex", "w") as f:
        for h in range(H):
            for s in range(STRIPS):
                # Extract strip: [8, 16]
                strip = img_perm[h, s*PP : (s+1)*PP, :]
                f.write(pack_strip(strip) + "\n")
    
    # 4. Dump Weights Hex
    # Single line containing all weights
    with open("weights.hex", "w") as f:
        f.write(pack_weights(weights_core.int()) + "\n")
        
    # 5. Dump Golden Output Hex (For verification script)
    # Output is [1, 16, 16, 16]. Permute to [H, W, C]
    out_perm = out_golden.squeeze(0).permute(1, 2, 0)
    
    with open("golden_output.txt", "w") as f:
        for h in range(H):
            for s in range(STRIPS):
                # The hardware outputs strips.
                # Strip shape: [8, 16] (8 pixels, 16 output channels)
                strip = out_perm[h, s*PP : (s+1)*PP, :]
                
                # We write raw values to text for easy parsing later
                for p in range(PP):
                    vals = [str(strip[p, c].item()) for c in range(OC)]
                    f.write(",".join(vals) + "\n")

    print("Files generated: image_input.hex, weights.hex, golden_output.txt")

if __name__ == "__main__":
    main()
