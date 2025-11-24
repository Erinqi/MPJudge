import torch
import torch.nn as nn
from PIL import Image


class AdaIN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = nn.Linear(d_model, d_model)
        self.bias = nn.Linear(d_model, d_model)

    def forward(self, x, cond):  # x: [B, N, d], cond: [B, d]
        mu_x = x.mean(dim=1, keepdim=True)
        std_x = x.std(dim=1, keepdim=True)
        scale = self.scale(cond).unsqueeze(1)  # [B, 1, d]
        bias = self.bias(cond).unsqueeze(1)
        x_norm = (x - mu_x) / (std_x + 1e-5)
        return x_norm * scale + bias

class Block(nn.Module):
    def __init__(self, d_model=768, nhead=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.adain1 = nn.LayerNorm(d_model)
        self.adain2 = AdaIN(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, music_emb):
        # Gated AdaIN 1
        x = self.adain1(x)

        attn_out, _ = self.cross_attn(x, x, x)
        x_a = x + attn_out

        # Gated AdaIN 2
        x_b = self.adain2(x_a, music_emb)
        res = (x_b - x_a).norm(dim=-1)
        # Feed Forward
        x = x_b + self.ffn(x_b)

        return x, res

class MyModel(nn.Module):
    def __init__(self, image_size=256, patch_size=16, num_layers=12, d_model=1024, nhead=8):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model))

        self.blocks = nn.ModuleList([
            Block(d_model, nhead) for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, 1)
        self.res = []

    def forward(self, img, music_emb):
        self.res = []
        B = img.shape[0]
        x = self.patch_embed(img).flatten(2).transpose(1, 2)  # [B, N, d]
        x = x + self.pos_embed[:, :x.size(1)]
        for blk in self.blocks:
            x, res = blk(x, music_emb)
            self.res.append(res)
        x_avg = x.mean(dim=1)  # global avg pool
        return self.head(x_avg), self.res  # [B, 1]

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0)  # [1, 3, H, W]
        return image_tensor


class Music(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),   # [B, 64, T/2, F/2]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # [B, 128, T/4, F/4]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# [B, 256, T/8, F/8]
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),# [B, 256, T/8, F/8]
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))  # ➜ [B, 256, 1, 1]
        )
        self.proj = nn.Linear(512, out_dim)  # ➜ [B, 768]

    def forward(self, x):  # x: [B, 1, T, F]
        x = self.encoder(x)             # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)       # [B, 256]
        return self.proj(x)      
