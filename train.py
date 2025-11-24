import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from dataset import PaintingMusicPairDataset
from model_adain import MyModel, CLIP, Music
from tqdm import tqdm

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = MyModel().to(device)
    model = DDP(model, device_ids=[local_rank])
    
    music_encoder = Music().to(device)
    music_encoder = DDP(music_encoder, device_ids=[local_rank])

    dataset = PaintingMusicPairDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_set, shuffle=True)
    val_sampler = DistributedSampler(val_set, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=256, sampler=train_sampler, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=256, sampler=val_sampler, num_workers=8)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(music_encoder.parameters()),
        lr=1e-5
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=200)
    loss_fn = torch.nn.L1Loss()

    if local_rank == 0:
        writer = SummaryWriter(log_dir="runs/painting_music_training")

    best_val_loss = 0.02
    
    global_train_step = 0
    global_val_step = 0 
    
    for epoch in range(200):
        train_sampler.set_epoch(epoch)

        if local_rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"\n[Epoch {epoch}] 🔁 Training... LR: {current_lr:.6f}")

        train_bar = tqdm(train_loader, disable=(local_rank != 0))
        for image_t, music, score in train_bar:
            
            image_t = image_t.to(device)
            music = music.to(device)
            score = score.to(device)

            music_feat = music_encoder(music)

            pred = model(image_t, music_feat).squeeze()
            loss = loss_fn(pred, score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_description(f"Loss: {loss.item():.4f}")
            
            if local_rank == 0:
                writer.add_scalar('Loss/Train', loss.item(), global_train_step)
                global_train_step += 1

        val_losses = []

        if local_rank == 0:
            print(f"[Epoch {epoch}] 🧪 Validating...")

        val_bar = tqdm(val_loader, disable=(local_rank != 0))
        with torch.no_grad():
            for image_t, music, score in val_bar:
                image_t = image_t.to(device)
                music = music.to(device)
                score = score.to(device)

                music_feat = music_encoder(music)

                pred = model(image_t, music_feat).squeeze()
                loss = loss_fn(pred, score)
                val_losses.append(loss.item())
                val_bar.set_description(f"Val Loss: {loss.item():.4f}")

                if local_rank == 0:
                    writer.add_scalar('Loss/Validation', loss.item(), global_val_step)
                    global_val_step += 1

        if local_rank == 0:
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"[Epoch {epoch}] ✅ Avg Val Loss: {avg_val_loss:.4f}")
            
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
            writer.add_scalar('Loss/Validation_Avg', avg_val_loss, epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.module.state_dict(), "best_model1.pt")
                torch.save(music_encoder.module.state_dict(), "best_music_encoder1.pt")
                print(f"[Epoch {epoch}] 💾 Best model saved with Val Loss: {best_val_loss:.4f}")
        
        scheduler.step()
    
    if local_rank == 0:
        writer.close()
    

if __name__ == "__main__":
    main()