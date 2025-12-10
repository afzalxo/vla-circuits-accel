import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

from models.gated_models import MoEBabyAIVLA


MODEL_PATH = "best_vla_moe_fast.pth"
VOCAB_PATH = "vocab.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOP_K = 32   
HIDDEN_DIM = 256 
STACK_FRAMES = 4 
DROPOUT_RATE = 0.5
NUM_EPISODES = 1000
MAX_STEPS = 64 

def evaluate():
    if not os.path.exists(VOCAB_PATH) or not os.path.exists(MODEL_PATH):
        print("Error: Model or Vocab not found.")
        return

    vocab = torch.load(VOCAB_PATH)
    vocab_size = len(vocab) + 1
    
    model = MoEBabyAIVLA(vocab_size).to(DEVICE)
    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    # Remove _orig_mod. prefix from keys
    for key in list(sd.keys()):
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod."):]
            sd[new_key] = sd.pop(key)
    model.load_state_dict(sd)
    model.eval()
    print(f"Sparse Model (Top-{TOP_K}) loaded successfully.")
    
    # Setup Env
    env = gym.make("BabyAI-GoToLocal-v0", render_mode="rgb_array")
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    success_count = 0
    print(f"\nStarting Evaluation on {NUM_EPISODES} episodes...")
    print("-" * 50)

    for i in range(NUM_EPISODES):
        obs, _ = env.reset()
        mission = env.unwrapped.mission
        
        text_ids = [vocab.get(w, 0) for w in mission.lower().split()]
        text_tensor = torch.tensor(text_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
        
        terminated = False
        truncated = False
        step_count = 0
        episode_reward = 0
        
        frame_buffer = [] 
        prev_action = 2 # Default Forward
        
        while not (terminated or truncated) and step_count < MAX_STEPS:
            curr_pil = Image.fromarray(obs)
            curr_tensor = transform(curr_pil).unsqueeze(0).to(DEVICE)
            
            frame_buffer.append(curr_tensor)
            if len(frame_buffer) > STACK_FRAMES:
                frame_buffer.pop(0)
            
            frames_to_stack = []
            while len(frames_to_stack) + len(frame_buffer) < STACK_FRAMES:
                frames_to_stack.append(frame_buffer[0])
            frames_to_stack.extend(frame_buffer)
            
            # Stack [1, 12, 56, 56]
            stacked_input = torch.cat(frames_to_stack, dim=1)
            
            prev_act_tensor = torch.tensor([prev_action], dtype=torch.long).to(DEVICE)
            
            with torch.no_grad():
                logits, _ = model(stacked_input, text_tensor, prev_act_tensor)
                action = torch.argmax(logits, dim=1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            prev_action = action
            
            if reward > 0:
                episode_reward = reward
                success_count += 1
                break
        
        status = "SUCCESS" if episode_reward > 0 else "FAIL"
        print(f"Episode {i+1:03d}: {status} | Steps: {step_count} | Goal: {mission}")

    print("-" * 50)
    success_rate = (success_count / NUM_EPISODES) * 100
    print(f"Final Success Rate: {success_count}/{NUM_EPISODES} ({success_rate:.1f}%)")

if __name__ == "__main__":
    evaluate()
