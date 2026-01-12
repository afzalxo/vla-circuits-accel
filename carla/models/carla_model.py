import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

class SimpleTokenizer:
    def __init__(self, texts):
        word_counts = Counter()
        for text in texts: word_counts.update(text.lower().split())
        self.vocab = {word: i+1 for i, (word, _) in enumerate(word_counts.items())}
        self.vocab_size = len(self.vocab) + 1
    def encode(self, text):
        return torch.tensor([self.vocab.get(w, 0) for w in text.lower().split()], dtype=torch.long)

class CarlaVLA(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512):
        super().__init__()
        
        # --- LANGUAGE ENCODER ---
        self.embedding = nn.EmbeddingBag(vocab_size, 64, mode='mean')
        
        # --- VISION BACKBONE (VGG-Style) ---
        # Input: 12 Channels (4 frames * 3 RGB)
        # We use MaxPool to downsample, keeping Conv stride=1 for dense feature extraction
        
        # Block 1: 12 -> 64 (Hardware: 1 -> 4 tiles)
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1, stride=2)
        
        # Block 2: 64 -> 128 (Hardware: 4 -> 8 tiles)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        
        # Block 3: 128 -> 128 (Hardware: 8 -> 8 tiles)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)
        
        # Block 4: 128 -> 256 (Hardware: 8 -> 16 tiles)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        
        # Block 5: 256 -> 256 (Hardware: 16 -> 16 tiles)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
        
        self.flatten = nn.Flatten()
        
        # Vision Dim: 256 channels * 4 * 4 spatial = 4096
        self.vision_dim = 4096
        
        # --- FUSION & CONTROL ---
        # Concatenate: Vision(4096) + Lang(64)
        self.fusion_fc1 = nn.Linear(self.vision_dim + 64, hidden_dim)
        self.fusion_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Output: 9 Discrete Actions
        self.action_head = nn.Linear(hidden_dim // 2, 9)

    def forward(self, img, text):
        # Vision
        x = F.relu(self.conv1(img))
        
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.conv3(x))
        
        x = F.relu(self.conv4(x))
        
        x = F.relu(self.conv5(x))
        
        vis_feat = self.flatten(x)
        
        # Language
        lang_feat = self.embedding(text)
        
        # Fusion
        fused = torch.cat([vis_feat, lang_feat], dim=1)
        x = F.relu(self.fusion_fc1(fused))
        x = F.relu(self.fusion_fc2(x))
        
        return self.action_head(x)

    def get_conv_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
