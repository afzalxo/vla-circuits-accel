import wandb
import pandas as pd
import numpy as np
import cv2
import os
import random
from tqdm import tqdm

wandb.Table.MAX_ARTIFACT_ROWS = 400000

DATASET_DIR = "carla_dataset_weather"
CSV_PATH = os.path.join(DATASET_DIR, "data.csv")
IMG_DIR = os.path.join(DATASET_DIR, "images")


SAMPLE_SIZE = 10 
# Project name in W&B
WANDB_PROJECT = "vla-accel"

def draw_overlay(image, row):
    """Draws steering/throttle bars and text on the image for quick debugging."""
    # Make copy to avoid modifying original
    img = image.copy()
    h, w = img.shape[:2]
    
    # 1. Text Info
    text = f"Cmd: {row['instruction']}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    vals = f"S:{row['steer']:.2f} T:{row['throttle']:.2f} B:{row['brake']:.2f}"
    cv2.putText(img, vals, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 2. Steering Bar (Bottom Center)
    center_x = w // 2
    bar_width = 100
    # Map -1..1 to pixels
    steer_x = int(center_x + (row['steer'] * bar_width))
    
    # Background bar
    cv2.rectangle(img, (center_x - bar_width, h - 30), (center_x + bar_width, h - 10), (50, 50, 50), -1)
    # Active steer bar (Orange)
    cv2.rectangle(img, (center_x, h - 30), (steer_x, h - 10), (0, 165, 255), -1)
    # Center line
    cv2.line(img, (center_x, h - 35), (center_x, h - 5), (255, 255, 255), 2)

    # 3. Throttle (Green) / Brake (Red) Indicators
    # Right side vertical bars
    t_height = int(row['throttle'] * 50)
    b_height = int(row['brake'] * 50)
    
    # Throttle
    cv2.rectangle(img, (w - 20, h - 10 - t_height), (w - 10, h - 10), (0, 255, 0), -1)
    # Brake
    cv2.rectangle(img, (w - 40, h - 10 - b_height), (w - 30, h - 10), (0, 0, 255), -1)

    return img

def main():
    print(f"Loading dataset from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        print("Error: CSV not found.")
        return

    df = pd.read_csv(CSV_PATH)
    df = df[df['is_train_candidate'] == 1].reset_index()
    print(f"Total Rows: {len(df)}")
    
    # Initialize W&B
    run = wandb.init(project=WANDB_PROJECT, name="dataset_inspection")
    
    # --- 1. MACRO STATISTICS (Histograms) ---
    print("Logging distributions...")
    
    # Log histograms to see imbalance
    data = [[s] for s in df['steer']]
    table = wandb.Table(data=data, columns=["steer"])
    wandb.log({
        # "steering_distribution": wandb.plot.histogram(table, "steer", title="Steering Distribution"),
        "steering_distribution": wandb.Histogram(df["steer"]),
        "throttle_distribution": wandb.Histogram(df['throttle']),
        "brake_distribution": wandb.Histogram(df['brake']),
        "instruction_counts": wandb.plot.bar(
            wandb.Table(data=[[k, v] for k, v in df['instruction'].value_counts().items()], columns=["cmd", "count"]), 
            "cmd", "count", title="Instruction Balance"
        )
    })

    # --- 2. VISUAL INSPECTION (Random Sample Table) ---
    print(f"Creating visual sample table ({SAMPLE_SIZE} images)...")
    
    # Create a W&B Table
    columns = ["image", "instruction", "steer", "throttle", "brake", "traj_id", "frame_idx"]
    vis_table = wandb.Table(columns=columns)
    
    # Sample random rows
    sample_indices = random.sample(range(len(df)), min(len(df), SAMPLE_SIZE))
    
    for idx in tqdm(sample_indices):
        row = df.iloc[idx]
        img_path = os.path.join(IMG_DIR, row['image_path'])
        
        if os.path.exists(img_path):
            # Load
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Draw Overlay
            debug_img = draw_overlay(img, row)
            
            # Convert to RGB for W&B
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            
            # Add to table
            vis_table.add_data(
                wandb.Image(debug_img),
                row['instruction'],
                row['steer'],
                row['throttle'],
                row['brake'],
                row['trajectory_id'],
                idx
            )
    
    wandb.log({"visual_samples": vis_table})

    # --- 3. TEMPORAL CHECK (Video of one trajectory) ---
    print("Generating trajectory video...")
    
    # Find a long trajectory
    traj_counts = df['trajectory_id'].value_counts()
    print("Trajectory lengths (top 5):")
    print(traj_counts.head())
    long_traj_id = traj_counts.idxmax()
    # long_traj_id = 17
    traj_df = df[df['trajectory_id'] == long_traj_id].sort_index()
    
    print(f"Visualizing Trajectory ID {long_traj_id} ({len(traj_df)} frames)...")
    
    video_frames = []
    # Limit video to 300 frames to keep upload size small
    limit = min(len(traj_df), 600)
    
    for i in range(limit):
        row = traj_df.iloc[i]
        img_path = os.path.join(IMG_DIR, row['image_path'])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            # Draw overlay
            img = draw_overlay(img, row)
            # Convert BGR to RGB (W&B expects RGB or CHW)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Transpose to CHW for wandb.Video
            img = np.transpose(img, (2, 0, 1))
            video_frames.append(img)

    if video_frames:
        video_array = np.stack(video_frames) # [T, C, H, W]
        wandb.log({"trajectory_video": wandb.Video(video_array, fps=20, format="mp4")})

    print("Done! Check your W&B dashboard.")
    run.finish()

if __name__ == "__main__":
    main()
