import os
import torch
from torch.utils.data import DataLoader
from model_adain import MyModel, Music
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import json
from torch.utils.data import Dataset
from torchvision import transforms


class PaintingMusicPairDataset(Dataset):
    def __init__(self):
        json_path = "test.json"

        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.music_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item["image_path"]).convert("RGB")
        image_t = self.image_transform(image)

        audio = Image.open(item["audio_path"]).convert("RGB")
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

def overlay_residual_colormap_on_image(
    residual,
    base_img,
    save_path,
    patch_H=16,
    patch_W=16,
    alpha=0.5,
    colormap='seismic'
):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    res_map = residual.detach().cpu().reshape(1, 1, patch_H, patch_W)

    res_map_up = torch.nn.functional.interpolate(res_map, size=(256, 256), mode='bilinear', align_corners=False)
    res_np = res_map_up.squeeze().numpy()

    res_np = res_np - res_np.min()
    res_np = res_np / (res_np.max() + 1e-8)

    cmap = cm.get_cmap(colormap)
    colored_map = cmap(res_np)[..., :3]
    colored_map = (colored_map * 255).astype(np.uint8)

    heatmap = Image.fromarray(colored_map).convert("RGBA")
    base_img = base_img.convert("RGBA").resize((256, 256))

    heatmap.putalpha(int(alpha * 255))

    overlayed = Image.alpha_composite(base_img, heatmap)
    overlayed.save(save_path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device("cuda")

    model = MyModel().to(device)
    music_encoder = Music().to(device)

    model_ckpt_path = "best_model1.pt"
    music_ckpt_path = "best_music_encoder1.pt"

    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    music_encoder.load_state_dict(torch.load(music_ckpt_path, map_location=device))

    dataset = PaintingMusicPairDataset()

    train_loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for idx, (image_t, music, score) in enumerate(train_loader):
        image_t = image_t.to(device)
        music = music.to(device)
        score = score.to(device)
        music_feat = music_encoder(music)

        with torch.no_grad():
            pred, res_list = model(image_t, None, music_feat)
            print(f"predicted score: {pred.item():.2f}")

        image_np = image_t[0].detach().cpu() * 0.5 + 0.5
        image_np = (image_np * 255).clamp(0, 255).byte().numpy()
        image_np = image_np.transpose(1, 2, 0)
        base_img = Image.fromarray(image_np)
        base_img.save(f"residual_vis/image{idx}_input.png")

        for layer_id, res in enumerate(res_list):
            save_path = f"residual_vis/image{idx}_layer{layer_id}_overlay.png"
            res = sum(res) / len(res)
            overlay_residual_colormap_on_image(res, base_img, save_path, patch_H=16, patch_W=16, alpha=0.5)


if __name__ == "__main__":
    main()