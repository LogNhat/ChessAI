import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from env.dataset import get_dataloader
from model.alphazero_net_v1 import AlphaZeroNetV1

def train(data_dir='dataset/train_top_6m', epochs=5, batch_size=4096, num_workers=4, limit_positions=6_000_000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu True Finetuning trên thiết bị: {device} (Tesla M40 Optimized)")
    print(f"Dữ liệu: {data_dir} ({limit_positions} positions limit)")
    print(f"Batch size: {batch_size}, Num Workers: {num_workers}")
    print(f"Total True Finetune Epochs: {epochs}")
    
    # 1. Khởi tạo Model CHUẨN: AlphaZeroNetV1, 10 blocks, 192 filters
    model = AlphaZeroNetV1(num_blocks=10, num_channels=192).to(device)
    
    # Checkpoint mạnh nhất để lấy làm gốc
    best_v1_path = 'checkpoints/model_ep2_batch2000.pth'
    
    if not os.path.exists(best_v1_path):
        print(f"❌ LỖI KHÔNG TÌM THẤY {best_v1_path}. Cần file này để bắt đầu finetuning.")
        return

    # Tải siêu mẫu (super model) ep2_batch2000
    checkpoint = torch.load(best_v1_path, map_location=device)
    # Lấy state_dict gốc (có thể chênh lệch nếu checkpoint lưu cấu trúc khác, nhưng trường hợp này đảm bảo 100% giống nhau)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    old_epoch = checkpoint.get('epoch', '?')
    old_loss  = checkpoint.get('loss', '?')
    if isinstance(old_loss, float):
        old_loss_str = f"{old_loss:.4f}"
    else:
        old_loss_str = str(old_loss)
        
    print(f"✅ Đã tải thành công 100% trọng số từ: {best_v1_path}")
    print(f"   (Epoch cũ: {old_epoch}, Loss cũ: {old_loss_str})")
    
    # Phải reset lại optimizer và scheduler, vì chúng ta đang bắt đầu phase mới
    # Dùng LR nhỏ: 5e-5 để không phá vỡ liên kết có sẵn
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Dataloader
    dataloader = get_dataloader(
        data_dir, 
        batch_size, 
        num_workers=num_workers,
        limit_positions=limit_positions
    )
    
    writer = SummaryWriter('runs/chess_sl_m40_v1_finetune')
    os.makedirs('checkpoints', exist_ok=True)
    
    best_loss = float('inf')
    ckpt_path = 'checkpoints/v1_finetuned_best.pth'

    for epoch in range(epochs):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\n=== Bắt đầu V1-Finetune Epoch {epoch+1}/{epochs} (LR = {current_lr:.6f}) ===")
        
        model.train()
        total_p_loss = 0.0
        total_v_loss = 0.0
        total_loss = 0.0
        
        start_time = time.time()
        
        for batch_idx, (tensors, moves, values) in enumerate(dataloader):
            tensors, moves, values = tensors.to(device), moves.to(device), values.to(device)
            
            # Forward pass (FP32 native - Tốt nhất cho Maxwell M40)
            policy_logits, value_out = model(tensors)
            value_out = value_out.squeeze(1) 
            
            loss_p = policy_criterion(policy_logits, moves)
            loss_v = value_criterion(value_out, values)
            loss = loss_p + loss_v
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_p_loss += loss_p.item()
            total_v_loss += loss_v.item()
            total_loss += loss.item()
            
            # Save periodic backups
            if batch_idx > 0 and batch_idx % 2000 == 0:
                print(f"Đang lưu Backup Checkpoint tại Batch {batch_idx}...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': total_loss / (batch_idx + 1),
                }, f'checkpoints/v1_finetuned_ep{epoch+1}_batch{batch_idx}.pth')

            if batch_idx % 50 == 0:
                cur_lr_log = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | "
                      f"P_Loss: {loss_p.item():.4f} | V_Loss: {loss_v.item():.4f} | "
                      f"Total: {loss.item():.4f} | LR: {cur_lr_log:.2e}")
                
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
        
        scheduler.step()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"=> Mức loss tốt nhất ({best_loss:.4f}), lưu vào {ckpt_path}...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, ckpt_path)
            
        # Luôn lưu checkpoint theo epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, f'checkpoints/v1_finetuned_ep{epoch+1}.pth')
        
    writer.close()
    print(f"\n🎉 True Finetuning 5 epochs hoàn thành! Checkpoint xuất sắc nhất nằm tại: {ckpt_path}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    train(epochs=5, batch_size=4096, num_workers=4, limit_positions=6_000_000)
