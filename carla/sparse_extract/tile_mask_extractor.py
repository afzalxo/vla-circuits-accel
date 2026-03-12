import torch
import numpy as np

def extract_tile_masks(model, oc_par=16, target_sparsity_percent=50.0):
    target_layers = model.get_conv_layers()
    all_tile_scores =[]
    layer_tiles = {}

    for i, layer in enumerate(target_layers):
        weight = layer.weight.detach()
        OC = weight.shape[0]
        num_tiles = OC // oc_par
        
        blocks = weight.view(num_tiles, oc_par, -1)
        # Calculate L1 magnitude of each tile
        tile_scores = torch.mean(torch.abs(blocks), dim=(1, 2)).cpu().numpy()
        layer_tiles[f'conv{i+1}'] = tile_scores
        all_tile_scores.extend(tile_scores)

    dynamic_threshold = np.percentile(all_tile_scores, target_sparsity_percent)
    print(f"--- N-ISA Compiler: Extracting '{target_sparsity_percent}% Sparsity' Circuit ---")
    print(f"Dynamic Global Threshold determined: {dynamic_threshold:.6f}")

    circuit_masks = {}
    total_tiles = len(all_tile_scores)
    active_tiles = 0
    
    for i, layer in enumerate(target_layers):
        scores = layer_tiles[f'conv{i+1}']
        
        # 1 if score > threshold, else 0
        mask = (scores > dynamic_threshold).astype(int).tolist()
        if sum(mask) == 0:
            best_tile_idx = np.argmax(scores)
            mask[best_tile_idx] = 1
            print(f"  [!] Connectivity Safeguard Triggered for Conv{i+1}")
        circuit_masks[f'conv{i+1}'] = mask
        
        l_total = len(mask)
        l_active = sum(mask)
        active_tiles += l_active
        
        mask_str = "".join([str(b) for b in mask])
        print(f"Conv{i+1:<2} ({l_total*oc_par:>4} ch) | Mask: {mask_str} | Active: {l_active}/{l_total}")
        
    actual_sparsity = 100 * (1 - (active_tiles / total_tiles))
    print("-" * 60)
    print(f"Total Tiles: {total_tiles} | Active Tiles: {active_tiles} | Actual Sparsity: {actual_sparsity:.1f}%")
    
    return circuit_masks, dynamic_threshold
