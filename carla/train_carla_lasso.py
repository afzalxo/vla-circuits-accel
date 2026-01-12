import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from models.carla_model import CarlaVLA, SimpleTokenizer

DATASET_DIR = "carla_vla_dataset_hd"
CSV_PATH = os.path.join(DATASET_DIR, "data.csv")
IMG_DIR = os.path.join(DATASET_DIR, "images")

BATCH_SIZE = 64
LEARNING_RATE = 0.0005 # Lower LR for VGG
EPOCHS = 30
LASSO_LAMBDA = 0.0001  # Start small, CARLA is harder to learn than BabyAI
BLOCK_SIZE = 16
STACK_FRAMES = 4
IMG_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CarlaDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            # Normalize for CARLA (approximate mean/std)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Pre-load instructions to save time
        self.encoded_instructions = [self.tokenizer.encode(i) for i in self.data['instruction']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Frame Stacking
        frames = []
        for i in range(STACK_FRAMES):
            # Simple lookback, clamping to 0
            target_idx = max(0, idx - i)
            
            # Check for episode discontinuity (heuristic: large jump in image index or filename)
            # Since we don't have episode IDs in the simple collector, we assume continuity
            # unless the image filename jumps significantly, but for now simple clamping is fine.
            
            img_name = self.data.iloc[target_idx]['image_path']
            img_path = os.path.join(self.img_dir, img_name)
            
            try:
                with Image.open(img_path) as img:
                    frames.insert(0, self.transform(img))
            except:
                # Fallback if file read fails
                frames.insert(0, torch.zeros(3, IMG_SIZE, IMG_SIZE))

        img_stack = torch.cat(frames, dim=0) # [12, 128, 128]
        text = self.encoded_instructions[idx]
        action = torch.tensor(self.data.iloc[idx]['action'], dtype=torch.long)
        
        return img_stack, text, action

def collate_fn(batch):
    imgs, texts, acts = zip(*batch)
    imgs = torch.stack(imgs)
    acts = torch.stack(acts)
    max_len = max([len(t) for t in texts])
    padded_texts = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts): padded_texts[i, :len(t)] = t
    return imgs, padded_texts, acts

def compute_block_lasso(model):
    loss = 0.0
    for layer in model.get_conv_layers():
        w = layer.weight
        out_c, in_c, k, k = w.shape
        if out_c % BLOCK_SIZE != 0 or in_c % BLOCK_SIZE != 0: continue
        
        w_view = w.view(out_c // BLOCK_SIZE, BLOCK_SIZE, in_c // BLOCK_SIZE, BLOCK_SIZE, k, k)
        w_perm = w_view.permute(0, 2, 1, 3, 4, 5)
        norms = torch.sqrt(torch.sum(w_perm ** 2, dim=(2, 3, 4, 5)))
        loss += torch.sum(norms)
    return loss

def train():
    print("Loading Data...")
    df = pd.read_csv(CSV_PATH)
    tokenizer = SimpleTokenizer(df['instruction'].unique())
    
    dataset = CarlaDataset(CSV_PATH, IMG_DIR, tokenizer)
    
    # 80/20 Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    model = CarlaVLA(tokenizer.vocab_size).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    print(f"Starting Training (Lasso={LASSO_LAMBDA})...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, texts, acts in train_loader:
            imgs, texts, acts = imgs.to(DEVICE), texts.to(DEVICE), acts.to(DEVICE)
            
            optimizer.zero_grad()
            with autocast():
                logits = model(imgs, texts)
                task_loss = criterion(logits, acts)
                lasso = compute_block_lasso(model)
                loss = task_loss + (LASSO_LAMBDA * lasso)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == acts).sum().item()
            total += acts.size(0)
            
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, texts, acts in val_loader:
                imgs, texts, acts = imgs.to(DEVICE), texts.to(DEVICE), acts.to(DEVICE)
                with autocast():
                    logits = model(imgs, texts)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == acts).sum().item()
                val_total += acts.size(0)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
    torch.save(model.state_dict(), "carla_vla_lasso.pth")
    print("Model Saved.")

if __name__ == "__main__":
    train()
