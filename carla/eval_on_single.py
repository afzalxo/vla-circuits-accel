import sys
import os
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

DATASET_DIR = "carla_dataset_base"
CSV_PATH = os.path.join(DATASET_DIR, "data.csv")
IMG_DIR = os.path.join(DATASET_DIR, "images")

# A100 Optimization
BATCH_SIZE = 1 
NUM_WORKERS = 1 # CPU workers for data augmentation
PREFETCH_FACTOR = 1

MAX_TOKEN_LEN = 5
STACK_FRAMES = 1
IMG_SIZE = 256

CONTROL_SCALE = 1.0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def command_to_branch_id(cmd):
    cmd = str(cmd).lower().strip()
    if "left" in cmd:
        if "change" in cmd: return 3 # Change Lane Left
        return 1 # Turn Left
    elif "right" in cmd:
        if "change" in cmd: return 4 # Change Lane Right
        return 2 # Turn Right
    elif "stop" in cmd or "red light" in cmd:
        return 5 # Stop at Red Light
    else:
        # Follow lane
        return 0


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
    cv2.line(img_bgr, start_pt, prd_steer_end, (0, 0, 255), 2) # Model Red

    bar_h = 10
    bar_y_start = 40
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
    cv2.putText(img_bgr, f"{model[1]:.2f}", (w - 130, bar_y_start), 0, 0.3, (255, 255, 255), 1)
    cv2.putText(img_bgr, f"{model[2]:.2f}", (w - 130, bar_y_start + dist_between), 0, 0.3, (255, 255, 255), 1)

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


class CarlaImageCache:
    def __init__(self, csv_path, img_dir, sample_idx=224):
        print(f"Loading Metadata from {csv_path}...")
        self.data = pd.read_csv(csv_path)
        def get_cam_code(path):
            if '_c.png' in path: return 0
            if '_l.png' in path: return 1
            if '_r.png' in path: return 2
            return 0
        self.data['cam_code'] = self.data['image_path'].apply(get_cam_code)
        self.data['virtual_traj_id'] = self.data['trajectory_id'] * 10 + self.data['cam_code']
        self.data = self.data.sort_values(['virtual_traj_id', 'image_path']).reset_index(drop=True)
        self.img_dir = img_dir
        self.tokenizer = SimpleTokenizer(self.data['instruction'].astype(str).unique())
        self.encoded_instructions = [self.tokenizer.encode(str(i)) for i in self.data['instruction']]
        
        self.data = self.data.iloc[sample_idx:sample_idx+1, :].reset_index(drop=True)
        print(f"Loading {len(self.data)} images into RAM (uint8)...")
        print(self.data)
        self.images = {}
        for idx in tqdm(range(len(self.data))):
            img_name = self.data.iloc[idx]['image_path']
            img_path = os.path.join(self.img_dir, img_name)
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
        # current_traj = self.manager.data.iloc[idx]['trajectory_id']
        current_traj = self.manager.data.iloc[idx]['virtual_traj_id']
        do_flip = self.is_train and (random.random() < FLIP_PROB)
        
        frames = []
        frames.insert(0, self._get_raw(idx, False))

        img_stack = np.concatenate(frames, axis=2) 
        img_stack = np.transpose(img_stack, (2, 0, 1)) 

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
        if idx > 0 and self.manager.data.iloc[idx-1]['virtual_traj_id'] == current_traj:
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
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint for evaluation')
    parser.add_argument('--sample_idx', type=int, default=224, help='Use fast in-RAM dataset (requires more RAM)')
    args = parser.parse_args()

    model_path = args.model_path
    sample_idx = args.sample_idx

    print(f"Reading CSV from {CSV_PATH}...")
    full_df = pd.read_csv(CSV_PATH)
    full_df = full_df.iloc[sample_idx:sample_idx+1, :].reset_index(drop=True)
    print(full_df)

    model = CarlaVLA(STACK_FRAMES).to(DEVICE)

    sd = torch.load(model_path, map_location='cpu')
    new_sd = {}
    for k, v in sd.items():
        if 'bias' not in k:
            new_sd[k] = v
        
    model.eval()
    model.load_state_dict(new_sd, strict=True)

    valid_df = full_df
    # valid_df = valid_df.iloc[:500, :].reset_index(drop=True)
    
    train_indices = valid_df.index.tolist()
    
    print(f"Train Frames: {len(train_indices)}")
    
    print("Using Fast In-RAM Dataset...")
    cache_manager = CarlaImageCache(CSV_PATH, IMG_DIR, sample_idx=sample_idx)
    eval_ds = CarlaFastDataset(cache_manager, train_indices, is_train=False)
    
    train_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                              prefetch_factor=PREFETCH_FACTOR, collate_fn=collate_fn)
    
    augmenter = GPUAugmenter().to(DEVICE) 
    
    model.eval()

    for imgs, texts, branch_ids, prevs, targets in train_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        texts = texts.to(DEVICE, non_blocking=True)
        branch_ids = branch_ids.to(DEVICE, non_blocking=True)
        prevs = prevs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        
        imgs = augmenter(imgs, is_train=False)
        
        prevs = torch.zeros_like(prevs)
        
        print(imgs[0, 0, :15, :15])
        preds = model(imgs, prevs)
        print(preds)
        
 
if __name__ == "__main__":
    train()
