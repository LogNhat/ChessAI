"""
Chess Deep Q-Learning (DQN) Training Script.
Pure Reinforcement Learning from scratch (or baseline initialization).
Implements Parallel Vector Self-Play, Replay Buffer, and Negamax Q-learning for zero-sum games.
"""
import os
import time
import math
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import chess

# Local imports
from model.dqn_net import ChessDQN
from env.encoder import Encoder, MOVE_TO_INDEX, INDEX_TO_MOVE, NUM_MOVES

# ─── Material Values Helper ──────────────────────────────────────────────────
def get_material_value(board: chess.Board, color: chess.Color) -> float:
    """Computes total material value of a color on the board."""
    piece_values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0
    }
    val = 0.0
    for piece_type, weight in piece_values.items():
        val += len(board.pieces(piece_type, color)) * weight
    return val

def compute_reward(board_before: chess.Board, board_after: chess.Board, player_color: chess.Color) -> tuple[float, bool]:
    """
    Computes a dense reward for the player who just moved.
    Reward elements:
    - Checkmate: +10.0
    - Draw: -0.2 (slight penalty to encourage active play over draws)
    - Material Capture: change in material balance (our_material - opponent_material)
    - Step penalty: -0.01 per move to encourage fast wins
    """
    if board_after.is_game_over():
        if board_after.is_checkmate():
            return 10.0, True
        else:
            return -0.2, True  # Draw penalty
            
    # Material balance change
    bal_before = get_material_value(board_before, player_color) - get_material_value(board_before, not player_color)
    bal_after = get_material_value(board_after, player_color) - get_material_value(board_after, not player_color)
    
    material_diff = bal_after - bal_before
    step_penalty = -0.01
    
    return material_diff + step_penalty, False

# ─── Experience Replay Buffer ────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, next_legal_indices, done):
        self.buffer.append((state, action, reward, next_state, next_legal_indices, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, next_legals, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            next_legals,
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

# ─── Vectorized Self-Play Environment ──────────────────────────────────────────
class VectorDQNPlay:
    def __init__(self, num_games: int):
        self.num_games = num_games
        self.boards = [chess.Board() for _ in range(num_games)]
        self.encoder = Encoder()
        self.game_lengths = [0] * num_games
        
    def step(self, model, epsilon, device, buffer):
        """
        Executes one step across all parallel environments.
        Collects experiences and saves them to the replay buffer.
        """
        model.eval()
        
        # Prepare state batch for active boards
        states = []
        for i in range(self.num_games):
            states.append(self.encoder.board_to_tensor(self.boards[i]))
            
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            q_values = model(states_tensor).cpu().numpy()
            
        stats = {"wins": 0, "draws": 0, "losses": 0, "moves": 0}
        
        for i in range(self.num_games):
            board = self.boards[i]
            legal_moves = list(board.legal_moves)
            
            # Find legal move indices
            legal_indices = []
            for move in legal_moves:
                idx = MOVE_TO_INDEX.get(move.uci())
                if idx is not None:
                    legal_indices.append(idx)
                    
            if not legal_indices:
                # No legal moves, reset board
                self.boards[i] = chess.Board()
                self.game_lengths[i] = 0
                continue
                
            # Epsilon-greedy move selection
            if random.random() < epsilon:
                chosen_idx = random.choice(legal_indices)
            else:
                # Mask illegal moves
                masked_q = np.full(NUM_MOVES, -1e9, dtype=np.float32)
                for idx in legal_indices:
                    masked_q[idx] = q_values[i][idx]
                chosen_idx = int(np.argmax(masked_q))
                
            chosen_move = chess.Move.from_uci(INDEX_TO_MOVE[chosen_idx])
            
            # Step the board
            board_before = board.copy()
            player_color = board.turn
            board.push(chosen_move)
            self.game_lengths[i] += 1
            
            # Calculate reward
            reward, done = compute_reward(board_before, board, player_color)
            
            # Get next state legal indices
            next_legal_indices = []
            if not done:
                for move in board.legal_moves:
                    idx = MOVE_TO_INDEX.get(move.uci())
                    if idx is not None:
                        next_legal_indices.append(idx)
                if not next_legal_indices:
                    done = True
                    reward = -0.2 # No legal moves represents draw or impasse
                    
            # Check maximum game length to prevent infinite games
            if self.game_lengths[i] >= 200:
                done = True
                reward = -0.2
                
            # Push to replay buffer
            state_tensor = self.encoder.board_to_tensor(board_before)
            next_state_tensor = self.encoder.board_to_tensor(board)
            buffer.push(state_tensor, chosen_idx, reward, next_state_tensor, next_legal_indices, done)
            
            # Handle termination & resetting
            if done:
                stats["moves"] += self.game_lengths[i]
                if board.is_checkmate():
                    stats["wins"] += 1
                else:
                    stats["draws"] += 1
                    
                # Reset board for a new game
                self.boards[i] = chess.Board()
                self.game_lengths[i] = 0
                
        return stats

# ─── Training Loop ────────────────────────────────────────────────────────────
def train_dqn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DQN] Starting Chess DQN RL Training on device: {device}")
    
    # 1. Hyperparameters
    NUM_PARALLEL_GAMES = 64
    BUFFER_CAPACITY = 100000
    BATCH_SIZE = 512
    GAMMA = 0.99
    LR = 2e-4
    WEIGHT_DECAY = 1e-5
    
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY = 20000 # Epsilon decays over this many step iterations
    
    TARGET_UPDATE_FREQ = 100 # How often to copy weights to target network
    STEPS_PER_ITERATION = 10  # Self-play steps per iteration
    GRADIENT_STEPS = 5        # Optimization steps per iteration
    
    # 2. Setup networks and buffer
    model = ChessDQN(num_blocks=6, num_channels=128).to(device)
    target_model = ChessDQN(num_blocks=6, num_channels=128).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    buffer = ReplayBuffer(BUFFER_CAPACITY)
    env = VectorDQNPlay(num_games=NUM_PARALLEL_GAMES)
    
    # Checkpoints directories
    os.makedirs('checkpoints', exist_ok=True)
    dqn_latest_path = 'checkpoints/dqn_latest.pth'
    dqn_best_path = 'checkpoints/dqn_best.pth'
    
    iteration = 0
    if os.path.exists(dqn_latest_path):
        print(f"[DQN] Resuming from checkpoint: {dqn_latest_path}")
        checkpoint = torch.load(dqn_latest_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        target_model.load_state_dict(model.state_dict())
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            pass
        iteration = checkpoint.get('iteration', 0)
        
    writer = SummaryWriter('runs/chess_dqn')
    
    # 3. Warming up replay buffer with some completely random moves
    print("[DQN] Warming up Replay Buffer with random moves...")
    while len(buffer) < BATCH_SIZE * 5:
        env.step(model, epsilon=1.0, device=device, buffer=buffer)
        print(f"  Buffer fill rate: {len(buffer)}/{BATCH_SIZE * 5}", end='\r')
    print(f"\n[DQN] Warmup completed! Starting reinforcement learning...")
    
    start_time = time.time()
    MAX_TIME = 2 * 3600  # 2 hours training budget
    
    # Track stats
    best_win_rate = 0.0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > MAX_TIME:
            print("[DQN] ⏱️ 2-hour time budget reached! Saving and exiting.")
            break
            
        iteration += 1
        
        # Calculate decayed epsilon
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1.0 * iteration / (EPSILON_DECAY / 5.0))
        
        # Step 1: Self-play collection
        total_wins = 0
        total_draws = 0
        total_moves = 0
        games_completed = 0
        
        for _ in range(STEPS_PER_ITERATION):
            stats = env.step(model, epsilon=epsilon, device=device, buffer=buffer)
            total_wins += stats["wins"]
            total_draws += stats["draws"]
            total_moves += stats["moves"]
            games_completed += stats["wins"] + stats["draws"]
            
        # Step 2: Policy optimization
        model.train()
        total_loss = 0.0
        
        for _ in range(GRADIENT_STEPS):
            states, actions, rewards, next_states, next_legals, dones = buffer.sample(BATCH_SIZE)
            
            states_t = torch.tensor(states).to(device)
            actions_t = torch.tensor(actions).to(device)
            rewards_t = torch.tensor(rewards).to(device)
            next_states_t = torch.tensor(next_states).to(device)
            dones_t = torch.tensor(dones).to(device)
            
            # Predict Q(s, a)
            q_values = model(states_t)
            state_action_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            
            # Predict target Q(s', a') with target network
            with torch.no_grad():
                next_q_values = target_model(next_states_t).cpu().numpy()
                
            # Negamax target value selection across batch
            # For zero-sum two-player games, Q_target(s, a) = r - gamma * max_a' Q(s', a')
            max_next_q_values = np.zeros(BATCH_SIZE, dtype=np.float32)
            for b_idx in range(BATCH_SIZE):
                if dones[b_idx]:
                    max_next_q_values[b_idx] = 0.0
                else:
                    legal_indices = next_legals[b_idx]
                    if not legal_indices:
                        max_next_q_values[b_idx] = -0.2  # Draw penalty if no moves
                    else:
                        max_next_q_values[b_idx] = np.max(next_q_values[b_idx][legal_indices])
                        
            max_next_q_t = torch.tensor(max_next_q_values).to(device)
            
            # Compute negamax target
            expected_state_action_values = rewards_t - GAMMA * max_next_q_t * (1.0 - dones_t)
            
            # Huber Loss (more robust to outliers than MSE)
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / GRADIENT_STEPS
        
        # Step 3: Copy to target network periodically
        if iteration % TARGET_UPDATE_FREQ == 0:
            target_model.load_state_dict(model.state_dict())
            print(f"[DQN Update] Synchronized target network parameters at iteration {iteration}")
            
        # Logging & reporting
        if iteration % 20 == 0:
            avg_game_length = total_moves / max(1, games_completed)
            win_rate = total_wins / max(1, games_completed)
            
            print(f"[DQN Iter {iteration} | {elapsed/3600:.2f}/2.0h] Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f} | Buffer: {len(buffer)} | WinRate: {win_rate:.1f}% | AvgLen: {avg_game_length:.1f}")
            
            writer.add_scalar('DQN/Loss', avg_loss, iteration)
            writer.add_scalar('DQN/Epsilon', epsilon, iteration)
            writer.add_scalar('DQN/WinRate', win_rate, iteration)
            writer.add_scalar('DQN/AvgGameLength', avg_game_length, iteration)
            writer.add_scalar('DQN/BufferLength', len(buffer), iteration)
            
            # Save latest checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iteration,
                'loss': avg_loss
            }, dqn_latest_path)
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'iteration': iteration,
                    'win_rate': win_rate
                }, dqn_best_path)
                print(f"[DQN Save] New best model saved to {dqn_best_path} with win rate {win_rate:.2f}")

    writer.close()
    print("[DQN] Training complete!")

if __name__ == "__main__":
    train_dqn()
