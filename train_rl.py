import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import chess
import numpy as np
from collections import deque
from model.alphazero_net_v1 import AlphaZeroNetV1
from env.encoder import Encoder, MOVE_TO_INDEX, NUM_MOVES

# ─── MCTS & Tự Chơi (Vectorized) ──────────────────────────────────────────────
class MCTSNode:
    def __init__(self, prior=0.0):
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.children = {} # action (chess.Move) -> MCTSNode
        
    def expand(self, legal_moves, action_probs, board=None):
        """Khởi tạo danh sách các nút con với Prior probability (P)"""
        for move in legal_moves:
            idx = MOVE_TO_INDEX.get(move.uci())
            if idx is not None:
                p = action_probs[idx]
                # Bơm P cục diện nếu đó là nước chiếu hết (Mate-in-1 Biasing)
                if board is not None:
                    board.push(move)
                    if board.is_checkmate():
                        p += 1000.0
                    board.pop()
                self.children[move] = MCTSNode(prior=p)

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self, c_puct=2.5):
        best_uct = -float('inf')
        best_child = None
        best_action = None
        
        sum_N = math.sqrt(self.N + 1)
        for action, child in self.children.items():
            u = c_puct * child.P * sum_N / (1 + child.N)
            uct = child.Q + u
            if uct > best_uct:
                best_uct = uct
                best_child = child
                best_action = action
        return best_action, best_child
    
    def backprop(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

def apply_dirichlet_noise(node, alpha=0.5, epsilon=0.3):
    """Bơm nhiễu Dirichlet để khuyến khích khám phá nước đi mới ở nút mốc gốc"""
    actions = list(node.children.keys())
    if not actions: return
    noise = np.random.dirichlet([alpha] * len(actions))
    for i, action in enumerate(actions):
        node.children[action].P = (1 - epsilon) * node.children[action].P + epsilon * noise[i]

def get_terminal_value(board):
    """Quy định Reward cốt lõi: Phạt cực nặng khi hòa trong MCTS."""
    res = board.result()
    if res == '1/2-1/2':
        return 5.0
    # Người đến lượt mắc kẹt (bị chiếu bí) => Họ thua => Value = -1.
    return -1.0 

def format_z_for_history(history, result_str):
    """Z target function: Hòa thì cả Trắng và Đen đều bị -1."""
    targets = []
    for (state, pi, color_turn) in history:
        if result_str == '1/2-1/2':
            z = -1.0 # Trắng hay Đen đều ăn trái đắng
        elif result_str == '1-0':
            z = 1.0 if color_turn == chess.WHITE else -1.0
        elif result_str == '0-1':
            z = 1.0 if color_turn == chess.BLACK else -1.0
        targets.append((state, pi, z))
    return targets

# ─── Quản Trị Tự Chơi Lô Song Song (Batched MCTS) ───────────────────────────
class VectorSelfPlay:
    def __init__(self, num_games=128):
        self.num_games = num_games
        self.boards = [chess.Board() for _ in range(num_games)]
        self.roots  = [MCTSNode() for _ in range(num_games)]
        self.histories = [[] for _ in range(num_games)]
        self.encoder = Encoder()
        
    def step_games(self, model, device, simulations=50):
        # 1. Bơm nhiễu cho rễ
        for i in range(self.num_games):
            if not self.boards[i].is_game_over() and len(self.roots[i].children) > 0:
                apply_dirichlet_noise(self.roots[i])

        # 2. Chạy Simulations
        for _ in range(simulations):
            leaf_tensors = []
            leaf_infos = [] # (board_idx, moves_taken, list(b.legal_moves))
            search_paths = []

            for i in range(self.num_games):
                if self.boards[i].is_game_over(): 
                    continue
                
                b = self.boards[i]
                node = self.roots[i]
                path = [node]
                moves_taken = []
                
                # Duyệt cây
                while not node.is_leaf():
                    action, child = node.select_child()
                    path.append(child)
                    b.push(action)
                    moves_taken.append(action)
                    node = child
                
                # Tại điểm lá
                is_draw_claimable = b.can_claim_draw() or b.is_repetition(2)
                if b.is_game_over() or is_draw_claimable:
                    if is_draw_claimable and not b.is_game_over():
                        val = 5.0
                    else:
                        val = get_terminal_value(b)
                    # Backprop ngay lập tức
                    for n in reversed(path):
                        val = -val * 0.99 # Đảo value từ góc nhìn đối thủ
                        n.backprop(val)
                else:
                    leaf_tensors.append(self.encoder.board_to_tensor(b))
                    # Lưu lại list moves để lát nữa expand còn xài checkmate-1
                    leaf_infos.append((i, moves_taken.copy(), list(b.legal_moves)))
                    search_paths.append(path)
                    
                # Hoàn trả board về gốc để không bị sai state cho vòng lặp tới!
                for _ in moves_taken:
                    b.pop()
                    
            if not leaf_tensors:
                break
                
            # Batch Inference liền mạch
            batch_t_in = torch.tensor(np.array(leaf_tensors), dtype=torch.float32).to(device)
            with torch.no_grad():
                logits, values = model(batch_t_in)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                values = values.squeeze(1).cpu().numpy()
                
            # Expand & Backprop
            for k, (board_idx, moves_taken, legal_moves) in enumerate(leaf_infos):
                b = self.boards[board_idx]
                
                # Tái thiết lập board tạm thời để expand
                for action in moves_taken:
                    b.push(action)
                    
                node = search_paths[k][-1]
                node.expand(legal_moves, probs[k], b)
                
                # Hoàn trả board
                for _ in moves_taken:
                    b.pop()
                
                # backprop
                val = values[k]
                for n in reversed(search_paths[k]):
                    val = -val * 0.99
                    n.backprop(val)
                    
        # 3. Chọn nước đi
        finished_games = []
        for i in range(self.num_games):
            if self.boards[i].is_game_over():
                continue
                
            root = self.roots[i]
            # Policy distribution $\pi$
            pi = np.zeros(NUM_MOVES, dtype=np.float32)
            total_n = sum(c.N for c in root.children.values())
            
            if total_n == 0:
                actions = list(self.boards[i].legal_moves)
                action = random.choice(actions)
                pi[MOVE_TO_INDEX[action.uci()]] = 1.0
            else:
                actions = []
                visits = []
                for a, c in root.children.items():
                    actions.append(a)
                    visits.append(c.N)
                    pi[MOVE_TO_INDEX[a.uci()]] = c.N / total_n
                
                # Khám phá ngẫu nhiên mở rộng thời lượng đầu game để tìm lối nhánh mới
                if len(self.histories[i]) < 120:
                    action = random.choices(actions, weights=visits, k=1)[0]
                else:
                    action = actions[np.argmax(visits)]
                    
            state_tensor = self.encoder.board_to_tensor(self.boards[i])
            self.histories[i].append((state_tensor, pi, self.boards[i].turn))
            
            self.boards[i].push(action)
            # Tiến góc rễ thay cho root mới
            if action in root.children:
                self.roots[i] = root.children[action]
            else:
                self.roots[i] = MCTSNode()
                
            # Xử lý kết thúc
            if self.boards[i].is_game_over() or len(self.histories[i]) > 300: # Cắt ván cờ quá dài
                res = self.boards[i].result() if self.boards[i].is_game_over() else '1/2-1/2'
                finished_games.extend(format_z_for_history(self.histories[i], res))
                
                # Reset board i
                self.boards[i] = chess.Board()
                self.roots[i] = MCTSNode()
                self.histories[i] = []
                
        return finished_games

# ─── Vòng lặp huấn luyện Chính (Manager) ─────────────────────────────────────
def run_rl_loop():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True # Tối ưu hóa Static Graph Convolutions cho GPU M40 Maxwell
    print(f"[RL] Khởi tạo hệ thống RL - Draw = Loss trên {device}")
    
    model = AlphaZeroNetV1(10, 192).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    rl_ckpt = 'checkpoints/rl_latest.pth'
    base_ckpt = 'checkpoints/v1_finetuned_best.pth'
    
    iteration = 0
    if os.path.exists(rl_ckpt):
        print(f"[RL] Đang Nối Tiếp tiến trình cũ từ {rl_ckpt}...")
        c = torch.load(rl_ckpt, map_location=device)
        model.load_state_dict(c['model_state_dict'])
        try:
            optimizer.load_state_dict(c['optimizer_state_dict'])
        except:
            pass
        iteration = c.get('iteration', 0)
    elif os.path.exists(base_ckpt):
        c = torch.load(base_ckpt, map_location=device)
        model.load_state_dict(c['model_state_dict'])
        print(f"[RL] Khởi tạo bắt đầu từ model Supervised {base_ckpt}")
    else:
        print("[RL] ❌ Không tìm thấy checkpoint nào!")
        return
        
    model.eval()
    # Replay Buffer
    replay_buffer = deque(maxlen=200000)
    
    # Ở server M40, chúng ta đánh 256 ván cùng lúc. (GPU 24GB thừa sức ôm batch size 256)
    # CPU xử lý 256 cây đồng thời trong 1 thread rất nhẹ do không nghẽn tree search python.
    NUM_PARALLEL_GAMES = 256
    SIMULATIONS_PER_MOVE = 200
    TRAIN_BATCH_SIZE = 4096
    GAMES_BEFORE_TRAIN = 2000 # Cứ thu thập khoảng 2000 nước đi thì Train 1 lần (dữ liệu chất lượng hơn).
    
    env = VectorSelfPlay(num_games=NUM_PARALLEL_GAMES)
    
    # Cần nạp mồi rễ bằng 1 nhịp neural ban đầu
    for i in range(NUM_PARALLEL_GAMES):
        legal = list(env.boards[i].legal_moves)
        tensor = torch.tensor(env.encoder.board_to_tensor(env.boards[i])).unsqueeze(0).to(device)
        with torch.no_grad():
            pol, _ = model(tensor)
        pol = F.softmax(pol, dim=1).cpu().numpy()[0]
        env.roots[i].expand(legal, pol, env.boards[i])

    writer = SummaryWriter('runs/chess_rl')
    
    start_time = time.time()
    MAX_TIME = 4 * 3600  # 4 tiếng
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > MAX_TIME:
            print("[RL] ⏱️ Đã hết 4 tiếng Train tối đa phục vụ nước rút. Dừng hệ thống an toàn!")
            break
            
        iteration += 1
        print(f"\n[RL Iteration {iteration} | {elapsed/3600:.2f}/4.0h] Bắt nạt AI tự đánh với chính mình...")
        model.eval()
        
        new_data = []
        # Chạy Tự Chơi tới khi đủ dữ liệu
        while len(new_data) < GAMES_BEFORE_TRAIN:
            finished = env.step_games(model, device, simulations=SIMULATIONS_PER_MOVE)
            new_data.extend(finished)
            # Log tiến trình
            if len(new_data) > 0 and len(finished) > 0:
                print(f"  -- Đã gom được {len(new_data)}/{GAMES_BEFORE_TRAIN} nước đi...")

        replay_buffer.extend(new_data)
        
        # Train Network!
        model.train()
        print(f"[RL Train] Tổng Buffer: {len(replay_buffer)}. Bắt đầu Cập nhật...")
        
        # Lấy ngẫu nhiên vài nghìn mẫu từ Buffer để đánh nhiều Epoch PPO-style
        train_samples = random.sample(replay_buffer, min(len(replay_buffer), TRAIN_BATCH_SIZE * 5))
        
        total_p = 0; total_v = 0; total_l = 0
        num_batches_processed = 0
        
        for epoch in range(3):
            random.shuffle(train_samples)
            batches = [train_samples[i:i+TRAIN_BATCH_SIZE] for i in range(0, len(train_samples), TRAIN_BATCH_SIZE)]
            
            for batch in batches:
                states = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(device)
                pi_ts  = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(device)
                z_ts   = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).to(device)
                
                p_logits, v_out = model(states)
                v_out = v_out.squeeze(1)
                
                # Policy loss (CrossEntropy against soft target pi)
                loss_p = -(pi_ts * F.log_softmax(p_logits, dim=1)).sum(dim=1).mean()
                # Value loss
                loss_v = F.mse_loss(v_out, z_ts)
                
                loss = loss_p + loss_v
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_p += loss_p.item()
                total_v += loss_v.item()
                total_l += loss.item()
                num_batches_processed += 1
                
        avg_l = total_l/num_batches_processed
        avg_p = total_p/num_batches_processed
        avg_v = total_v/num_batches_processed
        print(f"  => RL Cập nhật (3 epoch): Loss {avg_l:.4f} (P={avg_p:.4f}, V={avg_v:.4f})")
        
        # Ghi log Tensorboard
        writer.add_scalar('RL_Loss/Total', avg_l, iteration)
        writer.add_scalar('RL_Loss/Policy', avg_p, iteration)
        writer.add_scalar('RL_Loss/Value', avg_v, iteration)
        
        # Liên tục lưu Checkpoint mới để Worker đợt sau có Trực giác mới
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration,
            'rl_loss': avg_l,
        }, 'checkpoints/rl_latest.pth')
        
        print("[RL] Bắt đầu đồng bộ cho vòng đánh tự kỷ tiếp theo!")

if __name__ == "__main__":
    run_rl_loop()
