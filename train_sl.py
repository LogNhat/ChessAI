import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from env.dataset import get_dataloader
from model.alphazero_net import AlphaZeroNet

def train(data_dir='dataset/train_top_6m', epochs=20, batch_size=4096, num_workers=4, limit_positions=6_000_000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu Finetuning (tiếp tục học) trên thiết bị: {device} (Tesla M40 Optimized)")
    print(f"Batch size: {batch_size}, Max Data: {limit_positions} positions")
    print(f"Num Workers: {num_workers}, Total Finetune Epochs: {epochs}")
    
    # 1. Khởi tạo Model mới: 12 SE-ResBlocks, 192 filters
    # (tăng từ 10 lên 12, tích hợp SE Attention)
    model = AlphaZeroNet(num_blocks=12, num_channels=192).to(device)
    
    # 2. Losses và Optimizer
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Initial learning rate cho Finetuning Phase: nhỏ hơn để không phá vỡ weights đã học
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # CosineAnnealingLR: giảm LR mượt mà từ 1e-4 → eta_min=1e-7 qua 20 epochs
    # Phù hợp hơn MultiStepLR cho finetuning vì không bị nhảy cứng
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    
    # Lưu checkpoint finetuned riêng để không ghi đè model gốc
    ckpt_path     = 'checkpoints/finetuned_best.pth'
    # Checkpoint của model cũ (10 block, không SE) — dùng để transfer weights
    pretrain_path = 'checkpoints/best_model.pth'
    start_epoch = 0
    best_loss   = float('inf')

    # ── Ưu tiên 1: Load checkpoint finetuned (nếu đang resume finetuning) ──────
    if os.path.exists(ckpt_path):
        print(f"[RESUME] Tìm thấy finetuned checkpoint, đang tải {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss   = checkpoint.get('loss', float('inf'))
        print(f"[RESUME] Resume finetune từ epoch {start_epoch} (Best Loss: {best_loss:.4f})")

    # ── Ưu tiên 2: Transfer từ checkpoint cũ (khác architecture) ───────────────
    elif os.path.exists(pretrain_path):
        print(f"[TRANSFER] Tải pretrained weights từ {pretrain_path}...")
        checkpoint   = torch.load(pretrain_path, map_location=device)
        old_sd       = checkpoint['model_state_dict']
        new_sd       = model.state_dict()

        transferred, skipped_shape, skipped_missing = [], [], []

        for name, param in new_sd.items():
            if name in old_sd:
                if old_sd[name].shape == param.shape:
                    new_sd[name] = old_sd[name]          # ✅ khớp tên + shape
                    transferred.append(name)
                else:
                    skipped_shape.append(
                        f"  {name}: old {tuple(old_sd[name].shape)} != new {tuple(param.shape)}"
                    )
            else:
                skipped_missing.append(f"  {name}")      # layer mới hoàn toàn

        model.load_state_dict(new_sd)

        print(f"[TRANSFER] Transferred {len(transferred)} layers thành công.")
        if skipped_shape:
            print(f"[TRANSFER] Bỏ qua {len(skipped_shape)} layer lệch shape (giữ random init):")
            for s in skipped_shape:
                print(s)
        if skipped_missing:
            print(f"[TRANSFER] {len(skipped_missing)} layer mới hoàn toàn (giữ random init):")
            for s in skipped_missing:
                print(s)
        print("[TRANSFER] Bắt đầu finetune 20 epochs với SE-ResNet 12 blocks...")


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
            
            # Backward và optimize (FP32)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping: tránh exploding gradients trong giai đoạn finetuning
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                current_lr_log = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | "
                      f"P_Loss: {loss_p.item():.4f} | V_Loss: {loss_v.item():.4f} | "
                      f"Total: {loss.item():.4f} | LR: {current_lr_log:.2e}")
                
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
        
        # Cập nhật scheduler cuối mỗi epoch
        scheduler.step()
        
        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            print("=> Mức loss tốt nhất, đang lưu Finetuned Checkpoint...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'num_blocks': 12,
                'num_channels': 192,
            }, ckpt_path)
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, f'checkpoints/model_ep{epoch+1}.pth')
        
    writer.close()
    print("Finetuning 20 epochs hoàn thành! Checkpoint tốt nhất: checkpoints/finetuned_best.pth")

if __name__ == "__main__":
    # SE-ResNet 12 blocks, 192 filters.
    # Finetuning 20 epochs trên 6M nước đi chất lượng cao nhất (Elo >= 2985).
    # CosineAnnealingLR: lr từ 1e-4 → 1e-7
    train(epochs=20, batch_size=4096, num_workers=0, limit_positions=6_000_000)
