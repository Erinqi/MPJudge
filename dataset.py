import os
import json
import random
import torch
from torch.utils.data import Dataset
# import open_clip
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor

class PaintingMusicPairDataset(Dataset):
    def __init__(self):

        json_path = "painting_music_pairs_sampled_500k.json"

        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.music_transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item["image_path"]).convert("RGB")
        image_t = self.image_transform(image)

        audio = self.audio_loader(item["audio_path"])
        audio = self.image_transform(audio)

        score = torch.tensor(round(item["score"], 1), dtype=torch.float32)

        return image_t, audio, score

    def audio_loader(self, audio_path):

        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        mel_path = os.path.join("melspec_png", f"{base_name}.png")

        if not os.path.exists(mel_path):
            raise FileNotFoundError(f"Mel spectrogram image not found: {mel_path}")
        
        mel_image = Image.open(mel_path).convert("RGB")
        return mel_image