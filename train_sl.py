import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from env.dataset import get_dataloader
from model.alphazero_net import AlphaZeroNet

def train(data_dir='dataset/train_data', epochs=5, batch_size=4096, lr=0.001, num_workers=4, max_positions=20_000_000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu huấn luyện trên thiết bị: {device} (Tesla M40 Optimized)")
    print(f"Batch size: {batch_size}, Learning Rate: {lr}, Num Workers: {num_workers}, Epochs: {epochs}, Max Positions: {max_positions}")
    
    # 1. Khởi tạo Model (6 blocks, 128 filters phù hợp cho dữ liệu 20M)
    model = AlphaZeroNet(num_blocks=6, num_channels=128).to(device)
    
    ckpt_path = 'checkpoints/best_model.pth'
    start_epoch = 0
    if os.path.exists(ckpt_path):
        print(f"Đang tải Checkpoint từ {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Tải thành công!")

    # 2. Losses và Optimizer
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loại bỏ hoàn toàn AMP/Mixed Precision vì Maxwell M40 không có Tensor Cores hữu ích cho FP16
    
    # 3. Dataloader
    dataloader = get_dataloader(data_dir, batch_size, num_workers=num_workers, max_positions=max_positions)
    
    writer = SummaryWriter('runs/chess_sl_m40')
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    best_loss = float('inf')
    
    # Vòng lặp huấn luyện
    for epoch in range(start_epoch, epochs):
        model.train()
        total_p_loss = 0.0
        total_v_loss = 0.0
        total_loss = 0.0
        
        start_time = time.time()
        
        for batch_idx, (tensors, moves, values) in enumerate(dataloader):
            tensors, moves, values = tensors.to(device), moves.to(device), values.to(device)
            
            # Forward pass (FP32 native)
            policy_logits, value_out = model(tensors)
            
            value_out = value_out.squeeze(1) 
            
            loss_p = policy_criterion(policy_logits, moves)
            loss_v = value_criterion(value_out, values)
            loss = loss_p + loss_v
            
            # Backward and optimize (FP32)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            total_p_loss += loss_p.item()
            total_v_loss += loss_v.item()
            total_loss += loss.item()
            
            # Save periodic checkpoints to avoid data loss on 3 day run
            if batch_idx > 0 and batch_idx % 2000 == 0:
                print(f"Đang lưu Checkpoint dự phòng tại Batch {batch_idx}...")
                torch.save(model.state_dict(), f'checkpoints/model_ep{epoch+1}_batch{batch_idx}.pth')

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | "
                      f"P_Loss: {loss_p.item():.4f} | V_Loss: {loss_v.item():.4f} | "
                      f"Total: {loss.item():.4f}")
                
        # Cuối epoch
        avg_loss = total_loss / len(dataloader)
        avg_p_loss = total_p_loss / len(dataloader)
        avg_v_loss = total_v_loss / len(dataloader)
        
        epoch_time = time.time() - start_time
        print(f"--- Đã xong Epoch {epoch+1} trong {epoch_time:.2f}s ---")
        print(f"Avg Loss: {avg_loss:.4f} (Policy: {avg_p_loss:.4f}, Value: {avg_v_loss:.4f})")
        
        writer.add_scalar('Loss/Total', avg_loss, epoch)
        writer.add_scalar('Loss/Policy', avg_p_loss, epoch)
        writer.add_scalar('Loss/Value', avg_v_loss, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # Save checkpoints
        if avg_loss < best_loss:
            best_loss = avg_loss
            print("=> Mức loss tốt nhất, đang lưu Checkpoint...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, ckpt_path)
            
        torch.save(model.state_dict(), f'checkpoints/model_ep{epoch+1}.pth')

        scheduler.step()
        
    writer.close()
    print("Huấn luyện thành công!")

if __name__ == "__main__":
    # Batch size 4096 fits comfortably in 24GB VRAM for a 6-block 128-filter ResNet
    train(epochs=5, batch_size=4096, num_workers=0)
