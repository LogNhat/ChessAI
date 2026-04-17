import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from env.dataset import get_dataloader
from model.alphazero_net import AlphaZeroNet

def train(data_dir='dataset/train_data', epochs=45, batch_size=4096, num_workers=4, limit_positions=20_000_000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu huấn luyện quá trình 3 Phases trên thiết bị: {device} (Tesla M40 Optimized)")
    print(f"Batch size: {batch_size}, Max Data: {limit_positions} positions")
    print(f"Num Workers: {num_workers}, Total Epochs: {epochs}")
    
    # 1. Khởi tạo Model (10 blocks, 192 filters phù hợp cho thời hạn 3 ngày)
    model = AlphaZeroNet(num_blocks=10, num_channels=192).to(device)
    
    # 2. Losses và Optimizer
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Initial learning rate for Phase 1
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Phase Scheduler: 
    # Phase 1: epochs 0 -> 14 (lr=1e-3)
    # Phase 2: epochs 15 -> 29 (lr=1e-4)
    # Phase 3: epochs 30 -> 44 (lr=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=0.1)
    
    ckpt_path = 'checkpoints/best_model.pth'
    start_epoch = 0
    best_loss = float('inf')
    
    if os.path.exists(ckpt_path):
        print(f"Đang tải Checkpoint từ {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        print(f"Tải thành công! Resume từ epoch {start_epoch} (Best Loss: {best_loss:.4f})")

    # Loại bỏ hoàn toàn AMP/Mixed Precision vì Maxwell M40 không có Tensor Cores hữu ích cho FP16
    
    # 3. Dataloader
    dataloader = get_dataloader(
        data_dir, 
        batch_size, 
        num_workers=num_workers,
        limit_positions=limit_positions
    )
    
    writer = SummaryWriter('runs/chess_sl_m40')
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # Vòng lặp huấn luyện
    for epoch in range(start_epoch, epochs):
        current_lr = scheduler.get_last_lr()[0]
        # Xác định Phase hiện tại dựa vào epoch
        phase = 1 if epoch < 15 else (2 if epoch < 30 else 3)
        print(f"\n=== Bắt đầu Epoch {epoch+1}/{epochs} [Phase {phase}] (LR = {current_lr:.6f}) ===")
        
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
            
            # Save periodic checkpoints to avoid data loss on long runs
            if batch_idx > 0 and batch_idx % 2000 == 0:
                print(f"Đang lưu Checkpoint dự phòng tại Batch {batch_idx}...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': total_loss / (batch_idx + 1),
                }, f'checkpoints/model_ep{epoch+1}_batch{batch_idx}.pth')

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
        writer.add_scalar('LR', current_lr, epoch)
        
        # Cập nhật scheduler cuối mỗi epoch
        scheduler.step()
        
        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            print("=> Mức loss tốt nhất, đang lưu Checkpoint...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, ckpt_path)
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, f'checkpoints/model_ep{epoch+1}.pth')
        
    writer.close()
    print("Huấn luyện thành công toàn bộ 3 Phases!")

if __name__ == "__main__":
    # Batch size 4096 fits comfortably in 24GB VRAM.
    # Total epochs 45 separated into 3 phases by MultiStepLR.
    train(epochs=45, batch_size=4096, num_workers=0, limit_positions=20_000_000)
