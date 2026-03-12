import torch


def branched_loss(preds, targets, branch_ids):
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

    steer_loss_per_item = (error[:, 0] * 25.0 * steer_mag_weight) #.mean()

    # is_left_turn = (branch_ids == 1)
    # steer_loss_per_item[is_left_turn] = steer_loss_per_item[is_left_turn] * 3.0
    steer_loss = steer_loss_per_item.mean()

    # Accel Logic
    is_active_accel = torch.abs(targets[:, 1]) > 0.05
    accel_weight = torch.ones_like(targets[:, 1]) * 2.0
    accel_weight[is_active_accel] = 10.0
    
    accel_loss = (error[:, 1] * accel_weight).mean()
    
    return steer_loss + accel_loss, [steer_loss.item(), accel_loss.item()]


def compute_tile_group_lasso(model, oc_par=16):
    """
    Computes the Group Lasso penalty for hardware tiles.
    Forces entire blocks of 16 Output Channels to shrink to exactly zero.
    """
    lasso_penalty = 0.0
    
    # We only want to prune the vision backbone, not the specialized branches yet.
    # The branches are small and already task-specific.
    target_layers = model.get_conv_layers()
    
    for layer in target_layers:
        weight = layer.weight # Shape: [OC, IC, H, W]
        OC = weight.shape[0]
        
        # Only apply if OC is cleanly divisible by our Hardware Tile size
        if OC % oc_par == 0:
            num_tiles = OC // oc_par
            
            # Reshape weights into[num_tiles, oc_par, Everything Else]
            # e.g., 64 channels becomes 4 tiles of 16 channels
            blocks = weight.view(num_tiles, oc_par, -1)
            
            # Calculate the L2 norm of each tile (sqrt of sum of squares)
            # Adding 1e-8 prevents NaN gradients when a tile hits exactly zero
            tile_l2_norms = torch.sqrt(torch.sum(blocks**2, dim=(1, 2)) + 1e-8)
            
            # Sum the L2 norms to create the L1 penalty across groups (Group Lasso)
            lasso_penalty += torch.sum(tile_l2_norms)
            
    return lasso_penalty
