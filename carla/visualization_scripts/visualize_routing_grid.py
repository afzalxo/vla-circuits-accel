import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from models.carla_model import CarlaVLAHW_Gated


sd_path = '/home/eeafzal/vla-carla/model_checkpoints/carla_vla_furthest_driver_gated.pth'


def visualize_routing_grid_spaced(model, num_tasks=6):
    device = next(model.parameters()).device
    task_vec = torch.eye(num_tasks, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        all_pred_masks = model.gater(task_vec)
    split_masks = torch.split(all_pred_masks, model.predictable_tiles, dim=1)
    
    task_colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    task_names =["Follow Lane", "Turn Left", "Turn Right", "Change Lane Left", "Change Lane Right", "Stop"]
    inactive_color = '#e0e0e0'
    
    num_layers = len(model.conv_layers)
    max_oc_tiles = max([layer.out_channels // model.oc_par for layer in model.conv_layers])
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # --- Spacing Parameters ---
    x_gap = 0.25  # 25% empty space between layers horizontally
    y_gap = 0.05  # 5% empty space between tiles vertically
    
    for layer_idx, layer in enumerate(model.conv_layers):
        total_oc_tiles = layer.out_channels // model.oc_par
        
        # Total height allocated per tile (including the gap)
        tile_pitch = max_oc_tiles / total_oc_tiles
        # Actual drawn height of the rectangle
        drawn_height = tile_pitch * (1.0 - y_gap)
        
        # Width parameters
        layer_width = 1.0 - x_gap
        slot_width = layer_width / num_tasks

        dynamic_lw = max(0.3, 1.5 - (total_oc_tiles / max_oc_tiles))
        
        for tile_idx in range(total_oc_tiles):
            # Center the tile vertically within its pitch
            y_base = (tile_idx * tile_pitch) + (tile_pitch * y_gap / 2.0)
            x_base = layer_idx
            
            for t in range(num_tasks):
                is_active = False
                if tile_idx == 0:
                    is_active = True  # Polysemantic Core
                else:
                    pred = split_masks[layer_idx][t][tile_idx - 1].item()
                    if pred > 0.5:
                        is_active = True
                        
                # Make inactive blocks slightly transparent (alpha=0.4) so active colors pop
                color = task_colors[t] if is_active else inactive_color
                alpha = 1.0 if is_active else 0.4
                
                slot_x = x_base + (t * slot_width)
                rect = patches.Rectangle((slot_x, y_base), slot_width, drawn_height, 
                                         facecolor=color, alpha=alpha, edgecolor='white', linewidth=0.2)
                ax.add_patch(rect)

                border = patches.Rectangle((x_base, y_base), layer_width, drawn_height, fill=False, edgecolor='#333333', linewidth=dynamic_lw)
                ax.add_patch(border)
            
            # Draw border around the hardware tile block
            # border = patches.Rectangle((x_base, y_base), layer_width, drawn_height, 
                                       # fill=False, edgecolor='#555555', linewidth=1.0)
            # ax.add_patch(border)

    ax.set_xlim(0, num_layers)
    ax.set_ylim(0, max_oc_tiles)
    ax.invert_yaxis()
    
    # Center X-ticks on the drawn columns (x_base + half of layer_width)
    ax.set_xticks(np.arange(num_layers) + (1.0 - x_gap) / 2.0)
    ax.set_xticklabels([f"Conv{i+1}\n({layer.out_channels//model.oc_par} Tiles)" 
                        for i, layer in enumerate(model.conv_layers)], fontsize=12)
    
    ax.set_yticks([])
    ax.set_ylabel("Normalized Channel Capacity (100%)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Network Depth", fontsize=14, fontweight='bold')
    ax.set_title("N-ISA Hardware Allocation: Task-Specific Sub-Circuits", fontsize=18, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    legend_patches = [patches.Patch(color=task_colors[i], label=task_names[i]) for i in range(num_tasks)]
    legend_patches.append(patches.Patch(color=inactive_color, alpha=0.4, label="Inactive (Masked)"))
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("hardware_routing_spaced.png", dpi=300)
    plt.show()


import torch
import pandas as pd

def evaluate_computation_sparsity(model, img_size=256, num_tasks=6):
    device = next(model.parameters()).device
    task_vec = torch.eye(num_tasks, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        all_pred_masks = model.gater(task_vec) # Shape:[6, total_predictable_tiles]
        
    split_masks = torch.split(all_pred_masks, model.predictable_tiles, dim=1)
    
    # Use your actual task names if you have them
    task_names =["Follow Lane", "Turn Left", "Turn Right", "Change Lane Left", "Change Lane Right", "Stop"]
    
    results =[]
    
    for task_idx in range(num_tasks):
        total_macs = 0
        active_macs = 0
        total_params = 0
        active_params = 0
        
        # Conv1 input is 3 channels, padded to 1 IC tile (16 channels) in hardware
        total_ic_tiles = 1
        active_ic_tiles = 1
        
        current_spatial = img_size
        
        for i, layer in enumerate(model.conv_layers):
            # Handle tuple or int strides/kernel sizes
            stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
            k_size = layer.kernel_size[0] * layer.kernel_size[1] if isinstance(layer.kernel_size, tuple) else layer.kernel_size**2
            
            # Update spatial resolution
            current_spatial = current_spatial // stride
            spatial_pixels = current_spatial ** 2
            
            total_oc_tiles = layer.out_channels // model.oc_par
            
            # Active OC tiles: 1 (Always ON tile) + predicted mask sum
            pred_mask = (split_masks[i][task_idx] > 0.5).int()
            active_oc_tiles = 1 + pred_mask.sum().item()
            
            # Convert tiles back to padded channel widths
            ic_pad_tot = total_ic_tiles * model.oc_par
            oc_pad_tot = total_oc_tiles * model.oc_par
            ic_pad_act = active_ic_tiles * model.oc_par
            oc_pad_act = active_oc_tiles * model.oc_par
            
            # 1. MACs = Spatial_Pixels * IC * OC * Kernel^2
            layer_tot_macs = spatial_pixels * ic_pad_tot * oc_pad_tot * k_size
            layer_act_macs = spatial_pixels * ic_pad_act * oc_pad_act * k_size
            
            # 2. Parameters = (IC * OC * Kernel^2) + OC (bias)
            layer_tot_params = (ic_pad_tot * oc_pad_tot * k_size) + oc_pad_tot
            layer_act_params = (ic_pad_act * oc_pad_act * k_size) + oc_pad_act
            
            total_macs += layer_tot_macs
            active_macs += layer_act_macs
            total_params += layer_tot_params
            active_params += layer_act_params
            
            # The output of this layer becomes the input of the next
            total_ic_tiles = total_oc_tiles
            active_ic_tiles = active_oc_tiles
            
        mac_sparsity = (active_macs / total_macs) * 100
        param_sparsity = (active_params / total_params) * 100
        
        task_label = task_names[task_idx] if task_idx < len(task_names) else f"Task {task_idx}"
        
        results.append({
            "Task": task_label,
            "Active MACs": active_macs,
            "Total MACs": total_macs,
            "Active % (MACs)": mac_sparsity,
            "Active Params": active_params,
            "Total Params": total_params,
            "Active % (Params)": param_sparsity
        })
        
    df = pd.DataFrame(results)
    
    # Helper to format numbers clearly for papers (e.g., 1.23G for Billions, M for Millions)
    def format_num(num):
        if num >= 1e9: return f"{num/1e9:.2f}G"
        if num >= 1e6: return f"{num/1e6:.2f}M"
        if num >= 1e3: return f"{num/1e3:.2f}K"
        return str(num)
        
    # Create a formatted copy for printing
    df_formatted = df.copy()
    for col in ["Active MACs", "Total MACs", "Active Params", "Total Params"]:
        df_formatted[col] = df[col].apply(format_num)
        
    df_formatted["Active % (MACs)"] = df["Active % (MACs)"].apply(lambda x: f"{x:.1f}%")
    df_formatted["Active % (Params)"] = df["Active % (Params)"].apply(lambda x: f"{x:.1f}%")
    
    print("\n" + "="*95)
    print(" HARDWARE-ALIGNED COMPUTATION & PARAMETER SPARSITY (VLA Gated Conv Layers)")
    print("="*95)
    print(df_formatted.to_string(index=False, justify='center'))
    print("="*95 + "\n")
    
    # Return the raw dataframe in case you want to plot it later
    return df


def benchmark_gpu_latency(model, img_size=256, num_tasks=6, runs=100, warmup=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("Error: CUDA not available. Cannot profile GPU.")
        return

    model = model.to(device)
    model.eval()

    # Create dummy data for a single frame (Batch Size = 1 for Robotics)
    img = torch.randn(1, 3, img_size, img_size, device=device)
    prev_act = torch.randn(1, 3, device=device)
    
    print(f"--- GPU Latency Benchmarking (Batch Size 1) ---")
    
    # We will test Task 5 (Stop) which had 19.5% MAC sparsity, and Task 2 (Turn Right) with 70.3%
    tasks_to_test =[("Task 5 (Stop)", 5), ("Task 0 (Follow Lane)", 0)]
    
    for task_name, task_idx in tasks_to_test:
        # Create one-hot task vector
        task_vec = torch.zeros(1, num_tasks, device=device)
        task_vec[0, task_idx] = 1.0
        all_pred_masks = model.gater(task_vec)
        split_masks = torch.split(all_pred_masks, model.predictable_tiles, dim=1)

        # 1. Warm-up Phase
        print(f"Warming up GPU for {task_name}...")
        with torch.no_grad():
            for _ in range(warmup):
                _ = model.forward_latency_bench(img, prev_act, split_masks, task_idx)
                
        torch.cuda.synchronize() # Wait for warmup to finish

        # 2. Benchmark Phase
        start_events =[torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        end_events =[torch.cuda.Event(enable_timing=True) for _ in range(runs)]

        print(f"Running {runs} iterations...")
        with torch.no_grad():
            for i in range(runs):
                start_events[i].record()
                
                _ = model.forward_latency_bench(img, prev_act, split_masks, task_idx)
                
                end_events[i].record()

        torch.cuda.synchronize() # Wait for all runs to finish

        # Calculate average time in milliseconds
        times =[s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"GPU Latency - {task_name}: {avg_time:.3f} ms (std: {std_time:.3f} ms)")

task_dim = 6

model = CarlaVLAHW_Gated(task_dim=task_dim, oc_par=16)
sd = torch.load(sd_path, map_location='cpu')
model.load_state_dict(sd)

visualize_routing_grid_spaced(model, num_tasks=task_dim)

evaluate_computation_sparsity(model, img_size=256, num_tasks=task_dim)
benchmark_gpu_latency(model, img_size=256, num_tasks=task_dim, runs=1000, warmup=100)
