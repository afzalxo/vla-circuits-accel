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

class TextRouter(nn.Module):
    # Output: binary mask [Batch, Num_Layers, Num_Experts]
    def __init__(self, embed_dim, num_layers, num_experts, active_experts):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.k = active_experts
        
        # MLP to map Text Embedding -> Gating Logits
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_layers * num_experts)
        )

    def forward(self, text_embedding):
        logits = self.net(text_embedding)
        
        logits = logits.view(-1, self.num_layers, self.num_experts)
        
        # Top-K Selection (Hard Gating)
        # find the indices of the top K experts
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=2)
        
        # Zeros everywhere, Ones at the Top-K indices
        mask = torch.zeros_like(logits)
        mask.scatter_(2, top_k_indices, 1.0)
        
        # return both the mask (for hardware/forward) and logits (for training loss)
        return mask, logits

class GatedConv2d(nn.Module):
    """
    Conv layer split into parallel experts
    Hardware Implication: If mask[i] == 0, the FPGA skips this block entirely.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        
        # TODO: To keep parameter count fair vs the original, we divide channels by num_experts?
        self.experts = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            for _ in range(num_experts)
        ])

    def forward(self, x, layer_mask):
        output = 0
        
        # In a real FPGA, we would only iterate over active indices.
        # In PyTorch, we iterate all and multiply by 0 (mask).
        for i, expert in enumerate(self.experts):
            # reshape mask for broadcasting: [Batch, 1, 1, 1]
            gate = layer_mask[:, i].view(-1, 1, 1, 1)
            
            # If gate is 0, this adds 0 (simulating skipped compute)
            # Optimization: In inference, we can use 'if gate > 0' to actually skip but whatever for now
            out = expert(x)
            output = output + (out * gate)
            
        return output

class MoEBabyAIVLA(nn.Module):
    def __init__(self, vocab_size, num_experts=4, active_experts=1, top_k_logic=32):
        super().__init__()
        
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.top_k_logic = top_k_logic
        
        self.embedding = nn.EmbeddingBag(vocab_size, 32, mode='mean')
        
        self.router = TextRouter(embed_dim=32, num_layers=3, 
                                 num_experts=num_experts, active_experts=active_experts)
        
        self.logic_mask_net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256) # Output 256 scores for the MLP
        )

        # Layer 1
        self.conv1 = GatedConv2d(12, 32, 3, 2, 1, num_experts)
        self.relu1 = nn.ReLU()
        
        # Layer 2
        self.conv2 = GatedConv2d(32, 64, 3, 2, 1, num_experts)
        self.relu2 = nn.ReLU()
        
        # Layer 3
        self.conv3 = GatedConv2d(64, 128, 3, 2, 1, num_experts)
        self.relu3 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        # 128 x 7 x7 
        self.vision_dim = 6272
        self.action_emb = nn.Embedding(7, 16)
        self.fusion_linear = nn.Linear(self.vision_dim + 32 + 16, 256)
        self.dropout = nn.Dropout(0.5)

        self.action_head = nn.Linear(256, 5)

    def forward(self, img, text, prev_action):
        lang_feat = self.embedding(text)
        
        # masks shape: [Batch, 3, Num_Experts]
        masks, router_logits = self.router(lang_feat)
        
        # Pass corresponding mask to each layer
        x = self.conv1(img, masks[:, 0, :])
        x = self.relu1(x)
        
        x = self.conv2(x, masks[:, 1, :])
        x = self.relu2(x)
        
        x = self.conv3(x, masks[:, 2, :])
        x = self.relu3(x)
        
        vis_feat = self.flatten(x)
        
        act_feat = self.action_emb(prev_action)
        fused = torch.cat([vis_feat, lang_feat, act_feat], dim=1)
        hidden = self.fusion_linear(fused)
        
        logic_logits = self.logic_mask_net(lang_feat)
        _, top_i = torch.topk(logic_logits, k=self.top_k_logic, dim=1)
        logic_mask = torch.zeros_like(logic_logits)
        logic_mask.scatter_(1, top_i, 1.0)
        masked_hidden = hidden * logic_mask

        out = self.dropout(torch.relu(masked_hidden))

        logits = self.action_head(out)
        
        return logits, router_logits
