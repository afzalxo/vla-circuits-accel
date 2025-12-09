import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from minigrid.utils.baby_ai_bot import BabyAIBot
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

DATASET_DIR = "babyai_vla_dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images")
CSV_PATH = os.path.join(DATASET_DIR, "data.csv")
TOTAL_SAMPLES = 200000 

def collect_data():
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    # RGBImgPartialObsWrapper: Converts symbolic state to Pixels (Agent View)
    env = gym.make("BabyAI-GoToLocal-v0")
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    data_records = []
    sample_count = 0
    pbar = tqdm(total=TOTAL_SAMPLES)

    while sample_count < TOTAL_SAMPLES:
        # [56, 56, 3] RGB image
        obs, _ = env.reset()
        
        mission = env.unwrapped.mission
        
        bot = BabyAIBot(env.unwrapped)
        
        terminated = False
        truncated = False
        prev_action = 0
        
        while not (terminated or truncated):
            try:
                action = bot.replan()
            except Exception:
                break # failed to find path, skip episode

            img_filename = f"{sample_count:06d}.png"
            img_path = os.path.join(IMG_DIR, img_filename)
            
            Image.fromarray(obs).save(img_path)
            
            data_records.append({
                "image_index": sample_count,
                "image_path": img_filename,
                "instruction": mission,
                "action": action, # 0=Left, 1=Right, 2=Forward
                "episode_id": len(data_records)
            })
            
            prev_action = action

            obs, reward, terminated, truncated, info = env.step(action)
            
            sample_count += 1
            pbar.update(1)
            
            if sample_count >= TOTAL_SAMPLES:
                break

    df = pd.DataFrame(data_records)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df)} samples to {DATASET_DIR}")

if __name__ == "__main__":
    collect_data()
