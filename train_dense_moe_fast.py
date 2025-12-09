import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter
import time
from torch.cuda.amp import autocast, GradScaler # Mixed Precision
import wandb

from models.gated_models import MoEBabyAIVLA

DATASET_DIR = "babyai_vla_dataset"
CSV_PATH = os.path.join(DATASET_DIR, "data.csv")
IMG_DIR = os.path.join(DATASET_DIR, "images")

BATCH_SIZE = 2048
LEARNING_RATE = 0.002
EPOCHS = 150
NUM_WORKERS = 8

# Model Conf
HIDDEN_DIM = 256
TOP_K = 32
STACK_FRAMES = 4
DROPOUT_RATE = 0.5
NUM_EXPERTS = 4
ACTIVE_EXPERTS = 1

DEVICE = torch.device("cuda")

wandb.init(project="Dynamic Hardware Acceleration for Sparse Vision-Language Agents", name=f"moe_dense_{EPOCHS}ep_{STACK_FRAMES}frames")
wandb.config.update({
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "hidden_dim": HIDDEN_DIM,
    "dropout_rate": DROPOUT_RATE,
    "stack_frames": STACK_FRAMES,
    "num_experts": NUM_EXPERTS,
    "active_experts": ACTIVE_EXPERTS
})

# Ram cached dataset for speed
class FastBabyAIDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        
        self.transform = transforms.Compose([
            transforms.Resize((56, 56)),
            transforms.ToTensor()
        ])
        
        print("Loading dataset into RAM...")
        self.images = []
        self.instructions = []
        self.actions = []
        self.prev_actions = []
        
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            
            img_path = os.path.join(self.img_dir, row['image_path'])
            with Image.open(img_path) as img:
                self.images.append(self.transform(img))
            
            self.instructions.append(self.tokenizer.encode(row['instruction']))
            
            self.actions.append(row['action'])
            
            if idx > 0 and self.data.iloc[idx-1]['instruction'] == row['instruction']:
                self.prev_actions.append(self.data.iloc[idx-1]['action'])
            else:
                self.prev_actions.append(2) # Default Forward

        print(f"Loaded {len(self.data)} samples into RAM.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = []
        
        # We need to check episode boundaries using the pre-loaded data
        # Heuristic: Check if instruction matches previous indices
        current_instr_len = len(self.instructions[idx])
        
        for i in range(STACK_FRAMES):
            target_idx = idx - i
            
            valid = True
            if target_idx < 0: valid = False
            elif len(self.instructions[target_idx]) != current_instr_len: valid = False
            elif not torch.equal(self.instructions[target_idx], self.instructions[idx]): valid = False
            
            if valid:
                frames.insert(0, self.images[target_idx])
            else:
                if len(frames) > 0: frames.insert(0, frames[0])
                else: frames.insert(0, self.images[idx])

        img_tensor = torch.cat(frames, dim=0)
        return img_tensor, self.instructions[idx], torch.tensor(self.prev_actions[idx]), torch.tensor(self.actions[idx])

class SimpleTokenizer:
    def __init__(self, texts):
        word_counts = Counter()
        for text in texts: word_counts.update(text.lower().split())
        self.vocab = {word: i+1 for i, (word, _) in enumerate(word_counts.items())}
        self.vocab_size = len(self.vocab) + 1
    def encode(self, text):
        return torch.tensor([self.vocab.get(w, 0) for w in text.lower().split()], dtype=torch.long)

def collate_fn(batch):
    imgs, texts, prev_acts, actions = zip(*batch)
    imgs = torch.stack(imgs)
    prev_acts = torch.stack(prev_acts)
    actions = torch.stack(actions)
    max_len = max([len(t) for t in texts])
    padded_texts = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts): padded_texts[i, :len(t)] = t
    return imgs, padded_texts, prev_acts, actions

import torch.nn.functional as F

def train():
    print("Loading Data...")
    df = pd.read_csv(CSV_PATH)
    tokenizer = SimpleTokenizer(df['instruction'].unique())
    
    # class weights since there is class imbalance
    counts = df['action'].value_counts().sort_index()
    weights = 1.0 / counts
    if 2 in weights: weights = weights / weights[2]
    full_weights = torch.ones(5)
    for idx, w in weights.items(): 
        if idx < 5: full_weights[idx] = w
    class_weights = torch.clamp(full_weights, max=5.0).to(DEVICE).float()

    # load dset into RAM
    dataset = FastBabyAIDataset(CSV_PATH, IMG_DIR, tokenizer)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

    model = MoEBabyAIVLA(tokenizer.vocab_size, num_experts=NUM_EXPERTS, active_experts=ACTIVE_EXPERTS, top_k_logic=TOP_K).to(DEVICE)
    
    print("compiling...")
    model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler()

    best_acc = 0.0
    print(f"Starting Training (Batch={BATCH_SIZE})...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, texts, prev_acts, actions in train_loader:
            imgs, texts, prev_acts, actions = imgs.to(DEVICE), texts.to(DEVICE), prev_acts.to(DEVICE), actions.to(DEVICE)
            
            optimizer.zero_grad()
            
            with autocast():
                logits, router_logits = model(imgs, texts, prev_acts)
                
                task_loss = criterion(logits, actions)
                
                # load balancing loss: try to evenly distribute load across experts
                probs = F.softmax(router_logits, dim=2)
                mean_usage = torch.mean(probs, dim=0)
                target_usage = torch.full_like(mean_usage, 1.0 / NUM_EXPERTS)
                balance_loss = F.mse_loss(mean_usage, target_usage)
                
                loss = task_loss + (0.1 * balance_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += task_loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == actions).sum().item()
            total += actions.size(0)

        acc = correct / total
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, texts, prev_acts, actions in val_loader:
                imgs, texts, prev_acts, actions = imgs.to(DEVICE), texts.to(DEVICE), prev_acts.to(DEVICE), actions.to(DEVICE)
                with autocast():
                    logits, _ = model(imgs, texts, prev_acts)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == actions).sum().item()
                val_total += actions.size(0)
        val_acc = val_correct / val_total

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {acc:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_loader),
            "train_accuracy": acc,
            "val_accuracy": val_acc
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_vla_moe_fast.pth")
            wandb.save("best_vla_moe_fast.pth")
            print(f"New best model with acc {val_acc} saved.")

    torch.save(tokenizer.vocab, "vocab.pth")
    wandb.save("vocab.pth")
    print(f"Training complete... Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    train()
