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


class CarlaVLAHW(nn.Module):
        
    def __init__(self, stack_frames, hidden_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3 * stack_frames, 32, kernel_size=3, padding=1, stride=2, bias=False)
        # 128 x 128
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False)
        # 64 x 64
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False)
        # 32 x 32
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False)
        # 16 x 16
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False)
        # 8 x 8

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1, bias=False)
        # 8 x 8

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.vision_dim = 1024
        self.measure_proj = nn.Linear(3, 128) #, bias=False)
        self.fusion = nn.Linear(self.vision_dim + 128, hidden_dim) #, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.num_branches = 6
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256), # , bias=False),
                nn.ReLU(),
                nn.Linear(256, 256), #, bias=False),
                nn.ReLU(),
                nn.Linear(256, 2), #, bias=False),
                nn.Tanh()
            ) for _ in range(self.num_branches)
        ])

    def forward(self, img, prev_action):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        vis_feat = self.global_pool(x)
        vis_feat = vis_feat.view(vis_feat.size(0), -1)

        meas_feat = F.relu(self.measure_proj(prev_action))

        embedding = torch.cat([vis_feat, meas_feat], dim=1)
        embedding = F.relu(self.fusion(embedding))
        embedding = self.dropout(embedding)

        # Print the output of the first nn.Linear layer of the first branch for debugging
        # print(self.branches[0][1](self.branches[0][0](embedding)))

        outputs = []
        for branch in self.branches:
            outputs.append(branch(embedding))
            
        return torch.stack(outputs, dim=1)

    def get_conv_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]


class CarlaVLA(nn.Module):
    def __init__(self, vocab_size, stack_frames, hidden_dim=512):
        super().__init__()
        
        # self.action_proj = nn.Linear(3, 64, bias=False) 
        # self.embedding = nn.EmbeddingBag(vocab_size, 512, mode='mean')
        # self.lang_proj = nn.Linear(512, 512, bias=False)
        
        self.conv1 = nn.Conv2d(3 * stack_frames, 64, kernel_size=3, padding=1, stride=2, bias=False)
        # 128 x 128
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False)
        # 64 x 64
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False)
        # 32 x 32
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False)
        # 16 x 16
        
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2, bias=False)
        # 8 x 8

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2, bias=False)
        # 4 x 4
        self.conv7 = nn.Conv2d(1024, 256, kernel_size=3, padding=1, stride=1, bias=False)
        # 4 x 4

        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.vision_dim = 4096
        self.measure_proj = nn.Linear(3, 128)
        self.fusion = nn.Linear(self.vision_dim + 128, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.num_branches = 5
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 2),
                nn.Tanh()
            ) for _ in range(self.num_branches)
        ])

    def forward(self, img, text, prev_action):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        vis_feat = self.flatten(x)

        meas_feat = F.relu(self.measure_proj(prev_action))

        embedding = torch.cat([vis_feat, meas_feat], dim=1)
        embedding = F.relu(self.fusion(embedding))
        embedding = self.dropout(embedding)

        outputs = []
        for branch in self.branches:
            outputs.append(branch(embedding))
            
        return torch.stack(outputs, dim=1)

    def get_conv_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]

class CarlaVLASmall(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64):
        super().__init__()
        
        self.action_proj = nn.Linear(3, 64, bias=False) 
        self.embedding = nn.EmbeddingBag(vocab_size, 64, mode='mean')
        
        self.conv1 = nn.Conv2d(12, 16, kernel_size=3, padding=1, stride=2, bias=False)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=False)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.vision_dim = 64

        self.fusion_fc1 = nn.Linear(self.vision_dim + 64 + 64, hidden_dim, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.fusion_fc2 = nn.Linear(hidden_dim, hidden_dim // 2, bias=False)
        self.head_control = nn.Linear(hidden_dim // 2, 2, bias=False)

    def forward(self, img, text, prev_action):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.global_pool(x)
        vis_feat = x.view(x.size(0), -1)
        # Language
        lang_feat = self.embedding(text)
        act_feat = F.relu(self.action_proj(prev_action))
        # Fusion
        fused = torch.cat([vis_feat, lang_feat, act_feat], dim=1)
        x = F.relu(self.fusion_fc1(fused))
        x = self.dropout(x)
        x = F.relu(self.fusion_fc2(x))
        accel = torch.tanh(self.head_control(x))
        return accel

    def get_conv_layers(self):
        return [self.conv1, self.conv2, self.conv3]

'''
class CarlaVLA(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512):
        super().__init__()
        
        self.action_proj = nn.Linear(3, 64) 
        self.embedding = nn.EmbeddingBag(vocab_size, 64, mode='mean')
        
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm2d(1024)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.vision_dim = 1024
        # --- FUSION & CONTROL ---
        self.fusion_fc1 = nn.Linear(self.vision_dim + 64 + 64, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fusion_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.head_control = nn.Linear(hidden_dim // 2, 2)

    def forward(self, img, text, prev_action):
        x = F.relu(self.bn1(self.conv1(img)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        x = self.global_pool(x)
        vis_feat = x.view(x.size(0), -1)
        # Language
        lang_feat = self.embedding(text)
        act_feat = F.relu(self.action_proj(prev_action))
        # Fusion
        fused = torch.cat([vis_feat, lang_feat, act_feat], dim=1)
        x = F.relu(self.fusion_fc1(fused))
        x = self.dropout(x)
        x = F.relu(self.fusion_fc2(x))
        accel = torch.tanh(self.head_control(x))
        return accel

    def get_conv_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]
'''

class CarlaVLA128(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512):
        super().__init__()
        
        self.action_proj = nn.Linear(3, 64) 
        self.action_head = nn.Linear(hidden_dim // 2, 3)
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
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        
        # Block 4: 128 -> 256 (Hardware: 8 -> 16 tiles)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
        
        # Block 5: 256 -> 256 (Hardware: 16 -> 16 tiles)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)

        self.flatten = nn.Flatten()
        
        # Vision Dim: 256 channels * 4 * 4 spatial = 4096
        self.vision_dim = 4096
        # --- FUSION & CONTROL ---
        # Concatenate: Vision(4096) + Lang(64) + PrevRegress(64)
        self.fusion_fc1 = nn.Linear(self.vision_dim + 64 + 64, hidden_dim)
        self.fusion_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)

    def forward(self, img, text=None, prev_action=None):
        if prev_action is not None:
            act_feat = F.relu(self.action_proj(prev_action))
        # Vision
        x = F.relu(self.conv1(img))
        
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.conv3(x))
        
        x = F.relu(self.conv4(x))
        
        x = F.relu(self.conv5(x))

        vis_feat = self.flatten(x)
        
        if text is not None:
            # Language
            lang_feat = self.embedding(text)
        
        # Fusion
        if text is not None and prev_action is not None:
            fused = torch.cat([vis_feat, lang_feat, act_feat], dim=1)
        else:
            fused = vis_feat
        x = F.relu(self.fusion_fc1(fused))
        x = F.relu(self.fusion_fc2(x))
        
        return self.action_head(x)

    def get_conv_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
