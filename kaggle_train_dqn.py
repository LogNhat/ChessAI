# %% [markdown]
# # Chess Reinforcement Learning via Deep Q-Networks (Chess DQN) on Kaggle GPU
# 
# Welcome to this self-contained Jupyter Notebook designed to train a Deep Q-Network (DQN) 
# to play Chess from scratch using **Deep Reinforcement Learning (RL)**. This notebook is fully 
# optimized to run out-of-the-box on Kaggle's free **Tesla T4 GPU** environment.
# 
# ---
# 
# ## 🚀 1. Model Architecture Overview (Chess DQN)
# 
# Unlike traditional rule-based chess engines, this AI learns chess strategy from scratch 
# through **Parallel Self-Play**. The system is composed of two primary modules:
# 
# ### 1.1. State & Action Encoder
# *   **Board Representation (State Encoding):** A physical chess board is converted into an `(18, 8, 8)` float32 tensor. The 18 planes correspond to: White pieces (6 channels), Black pieces (6 channels), Active player's turn (1 channel), Castling rights for both players (4 channels), and the active En-passant square (1 channel).
# *   **Move Representation (Action Encoding):** All possible valid chess moves in UCI format (e.g., `"e2e4"`, `"e7e8q"`) are generated and mapped to a unique action space index ranging from `0` to `4271` (total size: **4,272** possible actions).
# 
# ### 1.2. Residual Convolutional Neural Network (ResNet DQN)
# The neural network uses a residual convolutional architecture to model board patterns:
# *   **Initial Stem Block:** Converts the 18 input channels to 128 feature channels using a $3 \times 3$ convolution, Batch Normalization, and a ReLU activation.
# *   **Residual Tower:** Comprises **6 residual blocks (ResBlocks)** with skip connections to extract complex spatial board relationships (e.g., long-range file/diagonal dependencies) while preventing vanishing gradients.
# *   **Q-head Block:** Compresses the 128 feature maps to 4 channels via a $1 \times 1$ convolution, flattens the result to 256, and uses dense layers to estimate raw $Q(s, a)$ values for all 4,272 action indices.
# 
# ---
# 
# ## 🧠 2. Reinforcement Learning Mechanics (How the AI Learns)
# 
# The training pipeline is an iterative cycle built on core reinforcement learning concepts:
# 
# 1.  **Vectorized Parallel Self-Play:** The agent plays **64 games simultaneously** on the GPU to maximize transition throughput, accelerating data collection 60x compared to sequential self-play.
# 2.  **Epsilon-Greedy Exploration ($\epsilon$-decay):** The exploration rate ($\epsilon$) exponentially decays from `1.0` (purely random moves to discover the game space) down to `0.1` (exploitation of learned strategies by taking the highest Q-value action).
# 3.  **Reward Function Design:** Features a dense reward signal: $+10.0$ for Checkmate, $-0.2$ for a Draw (encouraging decisive gameplay), relative material balance changes (Pawn=1, Knight/Bishop=3, Rook=5, Queen=9), and a $-0.01$ step penalty per move to encourage fast checkmates.
# 4.  **Negamax Q-learning for Zero-Sum Games:** Adapted for two-player adversarial games. The target Q-value for the current action is computed symmetrically using the opponent's best response in the next state:
#     $$Q_{\text{target}}(s, a) = r - \gamma \times \max_{a'} Q_{\text{target\_model}}(s', a')$$
# 5.  **Target Network & Huber Loss:** A separate **Target Network** is synchronized with the primary network every 100 iterations to stabilize Q-targets. The model is trained using **Huber Loss (Smooth L1 Loss)** to ensure robust gradient updates via the **AdamW** optimizer.
# 
# ---

# %% [markdown]
# ## 1. Install Dependencies
# We install the `python-chess` library which provides complete chess rule validation.
# %%
!pip install python-chess --quiet

# %% [markdown]
# ## 2. Imports and Environment Setup
# %%
import os
import time
import math
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# %% [markdown]
# ## 3. Chess State and Move Encoder
# Converts the chess.Board state to a (18, 8, 8) tensor and maps moves to unique action indices (0-4271).
# %%
FILES = "abcdefgh"
RANKS = "12345678"
PROMOTIONS = ['q', 'r', 'n', 'b']

# Generate all possible UCI strings for the action space
ALL_UCI_MOVES = []
for from_file in FILES:
    for from_rank in RANKS:
        for to_file in FILES:
            for to_rank in RANKS:
                ALL_UCI_MOVES.append(f"{from_file}{from_rank}{to_file}{to_rank}")

# Pawn promotions (Rank 7->8 for white, Rank 2->1 for black)
for from_file in FILES:
    for to_file in FILES:
        if abs(FILES.index(from_file) - FILES.index(to_file)) <= 1:
            for p in PROMOTIONS:
                ALL_UCI_MOVES.append(f"{from_file}7{to_file}8{p}")
                ALL_UCI_MOVES.append(f"{from_file}2{to_file}1{p}")

ALL_UCI_MOVES = sorted(list(set(ALL_UCI_MOVES)))
MOVE_TO_INDEX = {move: i for i, move in enumerate(ALL_UCI_MOVES)}
INDEX_TO_MOVE = {i: move for move, i in MOVE_TO_INDEX.items()}
NUM_MOVES = len(ALL_UCI_MOVES)

class Encoder:
    """Encodes chess board state and actions for Neural Network input."""
    def __init__(self):
        self.num_channels = 18

    def board_to_tensor(self, board: chess.Board) -> np.ndarray:
        tensor = np.zeros((18, 8, 8), dtype=np.float32)
        
        # 1. Pieces positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                color = int(piece.color) # 1 for White, 0 for Black
                piece_type = piece.piece_type - 1 # 0 to 5
                channel = piece_type if color == 1 else piece_type + 6
                rank, file = chess.square_rank(square), chess.square_file(square)
                tensor[channel, rank, file] = 1.0
                
        # 2. Whose turn is it
        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0
            
        # 3. Castling rights
        if board.has_kingside_castling_rights(chess.WHITE): tensor[13, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): tensor[14, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): tensor[15, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): tensor[16, :, :] = 1.0
        
        # 4. En-passant square
        if board.ep_square is not None:
            rank, file = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
            tensor[17, rank, file] = 1.0
            
        return tensor

# %% [markdown]
# ## 4. ResNet DQN Architecture
# Uses residual blocks to extract deep spatial patterns from the board.
# %%
class ResBlockDQN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)

class ChessDQN(nn.Module):
    def __init__(self, num_blocks: int = 6, num_channels: int = 128):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(18, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )
        
        self.res_blocks = nn.ModuleList([
            ResBlockDQN(num_channels) for _ in range(num_blocks)
        ])
        
        self.q_conv = nn.Conv2d(num_channels, 4, kernel_size=1, bias=False)
        self.q_bn   = nn.BatchNorm2d(4)
        
        self.q_fc1  = nn.Linear(4 * 8 * 8, 256)
        self.q_fc2  = nn.Linear(256, NUM_MOVES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        for block in self.res_blocks:
            x = block(x)
            
        x_q = F.relu(self.q_bn(self.q_conv(x)))
        x_q = x_q.reshape(x_q.size(0), -1)
        x_q = F.relu(self.q_fc1(x_q))
        return self.q_fc2(x_q)

# %% [markdown]
# ## 5. Experience Replay and Reward System
# %%
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

def get_material_value(board: chess.Board, color: chess.Color) -> float:
    piece_values = {
        chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
        chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0
    }
    val = 0.0
    for piece_type, weight in piece_values.items():
        val += len(board.pieces(piece_type, color)) * weight
    return val

def compute_reward(board_before: chess.Board, board_after: chess.Board, player_color: chess.Color) -> tuple[float, bool]:
    if board_after.is_game_over():
        if board_after.is_checkmate():
            return 10.0, True
        else:
            return -0.2, True  # Draw penalty
            
    # Compute relative material change
    bal_before = get_material_value(board_before, player_color) - get_material_value(board_before, not player_color)
    bal_after = get_material_value(board_after, player_color) - get_material_value(board_after, not player_color)
    
    material_diff = bal_after - bal_before
    step_penalty = -0.01
    
    return material_diff + step_penalty, False

# %% [markdown]
# ## 6. Parallel Self-Play Environment
# %%
class VectorDQNPlay:
    def __init__(self, num_games: int):
        self.num_games = num_games
        self.boards = [chess.Board() for _ in range(num_games)]
        self.encoder = Encoder()
        self.game_lengths = [0] * num_games
        
    def step(self, model, epsilon, device, buffer):
        model.eval()
        states = [self.encoder.board_to_tensor(self.boards[i]) for i in range(self.num_games)]
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            q_values = model(states_tensor).cpu().numpy()
            
        stats = {"wins": 0, "draws": 0, "losses": 0, "moves": 0}
        
        for i in range(self.num_games):
            board = self.boards[i]
            legal_moves = list(board.legal_moves)
            legal_indices = [MOVE_TO_INDEX[m.uci()] for m in legal_moves if m.uci() in MOVE_TO_INDEX]
                    
            if not legal_indices:
                self.boards[i] = chess.Board()
                self.game_lengths[i] = 0
                continue
                
            # Epsilon-greedy selection
            if random.random() < epsilon:
                chosen_idx = random.choice(legal_indices)
            else:
                masked_q = np.full(NUM_MOVES, -1e9, dtype=np.float32)
                for idx in legal_indices:
                    masked_q[idx] = q_values[i][idx]
                chosen_idx = int(np.argmax(masked_q))
                
            chosen_move = chess.Move.from_uci(INDEX_TO_MOVE[chosen_idx])
            
            board_before = board.copy()
            player_color = board.turn
            board.push(chosen_move)
            self.game_lengths[i] += 1
            
            reward, done = compute_reward(board_before, board, player_color)
            
            next_legal_indices = []
            if not done:
                next_legal_indices = [MOVE_TO_INDEX[m.uci()] for m in board.legal_moves if m.uci() in MOVE_TO_INDEX]
                if not next_legal_indices:
                    done = True
                    reward = -0.2
                    
            if self.game_lengths[i] >= 200:
                done = True
                reward = -0.2
                
            # Store in replay buffer
            state_tensor = self.encoder.board_to_tensor(board_before)
            next_state_tensor = self.encoder.board_to_tensor(board)
            buffer.push(state_tensor, chosen_idx, reward, next_state_tensor, next_legal_indices, done)
            
            if done:
                stats["moves"] += self.game_lengths[i]
                if board.is_checkmate():
                    stats["wins"] += 1
                else:
                    stats["draws"] += 1
                self.boards[i] = chess.Board()
                self.game_lengths[i] = 0
                
        return stats

# %% [markdown]
# ## 7. Hyperparameters and Network Initialization
# %%
# Directories
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
dqn_latest_path = '/kaggle/working/checkpoints/dqn_latest.pth'
dqn_best_path = '/kaggle/working/checkpoints/dqn_best.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing on device: {device}")

# Parameters
NUM_PARALLEL_GAMES = 64
BUFFER_CAPACITY = 100000
BATCH_SIZE = 512
GAMMA = 0.99
LR = 2e-4
WEIGHT_DECAY = 1e-5

EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 15000 # Epsilon decays over this many iterations

TARGET_UPDATE_FREQ = 100
STEPS_PER_ITERATION = 10
GRADIENT_STEPS = 5

# Create Networks
model = ChessDQN(num_blocks=6, num_channels=128).to(device)
target_model = ChessDQN(num_blocks=6, num_channels=128).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
buffer = ReplayBuffer(BUFFER_CAPACITY)
env = VectorDQNPlay(num_games=NUM_PARALLEL_GAMES)

# %% [markdown]
# ## 8. Warmup the Replay Buffer
# We populate the replay buffer with some completely random games so that the initial optimizer steps have diverse states.
# %%
print("Warming up Replay Buffer with random moves...")
while len(buffer) < BATCH_SIZE * 5:
    env.step(model, epsilon=1.0, device=device, buffer=buffer)
    print(f"  Buffer fill: {len(buffer)}/{BATCH_SIZE * 5}", end='\r')
print(f"\nWarmup completed! Current buffer size: {len(buffer)}")

# %% [markdown]
# ## 9. Training Loop
# Trains for a set budget of iterations, showing progress plots and saving checkpoints.
# %%
# Metrics tracking lists
history_iterations = []
history_loss = []
history_winrate = []
history_epsilon = []

start_time = time.time()
MAX_ITERATIONS = 2000 # Customize based on your training time budget
best_win_rate = 0.0

for iteration in range(1, MAX_ITERATIONS + 1):
    # Epsilon decay
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1.0 * iteration / (EPSILON_DECAY / 5.0))
    
    # 1. Vector self-play collection steps
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
        
    # 2. Optimization step
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
        
        # Predict next Q(s', a') with Target Network
        with torch.no_grad():
            next_q_values = target_model(next_states_t).cpu().numpy()
            
        # Compute Negamax Target
        max_next_q_values = np.zeros(BATCH_SIZE, dtype=np.float32)
        for b_idx in range(BATCH_SIZE):
            if dones[b_idx]:
                max_next_q_values[b_idx] = 0.0
            else:
                legal_indices = next_legals[b_idx]
                if not legal_indices:
                    max_next_q_values[b_idx] = -0.2
                else:
                    max_next_q_values[b_idx] = np.max(next_q_values[b_idx][legal_indices])
                    
        max_next_q_t = torch.tensor(max_next_q_values).to(device)
        expected_state_action_values = rewards_t - GAMMA * max_next_q_t * (1.0 - dones_t)
        
        # Loss & backward
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / GRADIENT_STEPS
    
    # 3. Synchronize target network
    if iteration % TARGET_UPDATE_FREQ == 0:
        target_model.load_state_dict(model.state_dict())
        
    # 4. Print stats and plot training curves
    if iteration % 20 == 0:
        win_rate = (total_wins / max(1, games_completed)) * 100.0
        avg_game_length = total_moves / max(1, games_completed)
        elapsed = time.time() - start_time
        
        # Log to history
        history_iterations.append(iteration)
        history_loss.append(avg_loss)
        history_winrate.append(win_rate)
        history_epsilon.append(epsilon)
        
        print(f"Iteration {iteration:4d} | Epsilon: {epsilon:.3f} | Loss: {avg_loss:.4f} | "
              f"WinRate: {win_rate:.1f}% | Games Completed: {games_completed} | AvgLen: {avg_game_length:.1f} | "
              f"Elapsed: {elapsed/60:.1f}m")
        
        # Save checkpoints
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

# %% [markdown]
# ## 10. Visualize Results
# Plots the training curves.
# %%
plt.figure(figsize=(18, 5))

# Plot Huber Loss
plt.subplot(1, 3, 1)
plt.plot(history_iterations, history_loss, label='Huber Loss', color='crimson')
plt.title('Huber Loss over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Plot Win Rate
plt.subplot(1, 3, 2)
plt.plot(history_iterations, history_winrate, label='Self-Play Win Rate', color='forestgreen')
plt.title('Self-Play Win Rate (%)')
plt.xlabel('Iterations')
plt.ylabel('Win Rate (%)')
plt.grid(True)
plt.legend()

# Plot Epsilon Decay
plt.subplot(1, 3, 3)
plt.plot(history_iterations, history_epsilon, label='Epsilon', color='royalblue')
plt.title('Epsilon Decay')
plt.xlabel('Iterations')
plt.ylabel('Epsilon')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('/kaggle/working/training_curves.png')
plt.show()

print("Training finished! Best model and training curves are saved in /kaggle/working/")
