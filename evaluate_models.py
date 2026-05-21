#!/usr/bin/env python3
"""
Chess AI Tournament & Evaluation Script
Allows all models to play against each other simultaneously,
ranks them using Elo ratings (Bradley-Terry maximum likelihood),
and logs the detailed tournament results to a file.
"""

import os
import sys
import time
import math
import random
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Add the parent directory to sys.path to import local modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import chess

# Local imports
from model.alphazero_net import AlphaZeroNet
from model.alphazero_net_v1 import AlphaZeroNetV1
from env.encoder import Encoder, MOVE_TO_INDEX, NUM_MOVES

# ─── Setup Logging ───────────────────────────────────────────────────────────
LOG_FILE = "checkpoints/tournament_results.log"
os.makedirs("checkpoints", exist_ok=True)

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("tournament")

# ─── MCTS & Evaluation Core ──────────────────────────────────────────────────

class MCTSNode:
    def __init__(self, prior=0.0):
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.children = {}

    def expand(self, legal_moves, action_probs, board=None):
        for move in legal_moves:
            idx = MOVE_TO_INDEX.get(move.uci())
            if idx is not None:
                p = action_probs[idx]
                if board is not None:
                    board.push(move)
                    if board.is_checkmate():
                        p += 1000.0
                    board.pop()
                self.children[move] = MCTSNode(prior=p)

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self):
        best_uct = -float('inf')
        best_child = None
        best_action = None
        sum_N = math.sqrt(self.N + 1)
        
        # AlphaZero dynamic PUCT formula
        c_init = 1.25
        c_base = 19652.0
        c_puct = c_init + math.log((self.N + c_base + 1) / c_base)
        
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


def get_terminal_value(board):
    res = board.result()
    if res == '1/2-1/2':
        return 5.0  # Punish draws heavily to encourage dynamic play
    return -1.0 


def apply_endgame_heuristic(b, v):
    if len(b.piece_map()) <= 5 and abs(v) > 0.5:
        if v < -0.5:
            loser = b.turn
            winner = not b.turn
            sign = -1
        else:
            loser = not b.turn
            winner = b.turn
            sign = 1
            
        l_king = b.king(loser)
        w_king = b.king(winner)
        if l_king is not None and w_king is not None:
            r_l = chess.square_rank(l_king)
            f_l = chess.square_file(l_king)
            edge = max(r_l, 7 - r_l) + max(f_l, 7 - f_l)
            
            r_w = chess.square_rank(w_king)
            f_w = chess.square_file(w_king)
            dist = max(abs(r_l - r_w), abs(f_l - f_w))
            
            h = (edge * 0.01) - (dist * 0.01)
            v += sign * h
            return max(-1.0, min(1.0, v))
    return v


def search_mcts_eval(board, net, device, encoder, num_simulations=50, temp=0.0):
    """
    Evaluates position using MCTS.
    If temp > 0, samples from visit counts (useful for early game variation).
    If temp == 0, selects deterministic best move with draw avoidance.
    """
    root = MCTSNode()
    
    tensor_t = torch.from_numpy(encoder.board_to_tensor(board)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = net(tensor_t)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
    root.expand(list(board.legal_moves), probs, board)
    
    for _ in range(num_simulations):
        b = board.copy()
        node = root
        path = [node]
        
        while not node.is_leaf():
            action, child = node.select_child()
            path.append(child)
            b.push(action)
            node = child
            
        is_draw_claimable = b.can_claim_draw() or b.is_repetition(2)
        if b.is_game_over() or is_draw_claimable:
            if is_draw_claimable and not b.is_game_over():
                v = 5.0
            else:
                v = get_terminal_value(b)
            for n in reversed(path):
                v = -v * 0.99
                n.backprop(v)
            continue
            
        tensor_t = torch.from_numpy(encoder.board_to_tensor(b)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = net(tensor_t)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            v = apply_endgame_heuristic(b, value.item())
            
        node.expand(list(b.legal_moves), probs, b)
        for n in reversed(path):
            v = -v * 0.99
            n.backprop(v)
            
    total_N = sum(c.N for c in root.children.values())
    if total_N == 0:
        return list(board.legal_moves)[0]

    # --- Draw Avoider Heuristic ---
    safe_actions = set()
    for m in board.legal_moves:
        board.push(m)
        is_draw = False
        if board.is_game_over() and board.result() == '1/2-1/2':
            is_draw = True
        elif board.can_claim_draw() or board.is_repetition(2) or board.halfmove_clock >= 100:
            is_draw = True
            
        if not is_draw:
            safe_actions.add(m)
        board.pop()

    candidate_moves = []
    for action, child in root.children.items():
        candidate_moves.append({
            "action": action,
            "N": child.N
        })
        
    candidate_moves.sort(key=lambda x: x["N"], reverse=True)
    
    if temp > 0.0:
        # Temperature-based sampling from safe candidates if available
        safe_candidates = [c for c in candidate_moves if c["action"] in safe_actions]
        if not safe_candidates:
            safe_candidates = candidate_moves
            
        visits = np.array([c["N"] for c in safe_candidates], dtype=np.float32)
        # Apply temperature
        visits = visits ** (1.0 / temp)
        sum_visits = visits.sum()
        
        if sum_visits == 0:
            probs = np.ones(len(safe_candidates)) / len(safe_candidates)
        else:
            probs = visits / sum_visits
            
        chosen = np.random.choice(safe_candidates, p=probs)
        return chosen["action"]
    else:
        # Deterministic choice favoring safe actions
        for cand in candidate_moves:
            if cand["action"] in safe_actions:
                return cand["action"]
        return candidate_moves[0]["action"]


def play_direct_policy(board, net, device, encoder, temp=0.0):
    """
    Quickly selects a move directly from the policy network output (no MCTS).
    """
    legal = list(board.legal_moves)
    if not legal:
        return None
        
    tensor_t = torch.from_numpy(encoder.board_to_tensor(board)).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, _ = net(tensor_t)
        
    logits = policy_logits[0].cpu().numpy()
    
    legal_indices = []
    legal_moves_map = {}
    for mv in legal:
        idx = MOVE_TO_INDEX.get(mv.uci())
        if idx is not None:
            legal_indices.append(idx)
            legal_moves_map[idx] = mv
            
    if not legal_indices:
        return random.choice(legal)
        
    masked = np.full(NUM_MOVES, -1e9, dtype=np.float32)
    for idx in legal_indices:
        masked[idx] = logits[idx]
        
    if temp > 0.0:
        ll = masked[legal_indices] / temp
        ll -= ll.max()
        probs = np.exp(ll)
        probs /= probs.sum()
        chosen_idx = np.random.choice(len(legal_indices), p=probs)
        return legal_moves_map[legal_indices[chosen_idx]]
    else:
        chosen_idx = int(np.argmax(masked[legal_indices]))
        return legal_moves_map[legal_indices[chosen_idx]]


# ─── Single Game Execution ───────────────────────────────────────────────────

def play_game(white_name, white_net, black_name, black_net, device, encoder,
              use_mcts=True, simulations=50, max_moves=150, temp_moves=6):
    """
    Plays a single chess game between two models.
    Returns: (white_name, black_name, result, termination, num_moves)
    result is '1-0' (White wins), '0-1' (Black wins), or '1/2-1/2' (Draw)
    """
    board = chess.Board()
    move_count = 0
    
    while not board.is_game_over() and move_count < max_moves:
        # Determine temperature to use
        # The first `temp_moves` plies have random sampling for opening variety
        current_temp = 1.0 if move_count < temp_moves else 0.0
        
        if board.turn == chess.WHITE:
            net = white_net
        else:
            net = black_net
            
        if use_mcts:
            move = search_mcts_eval(board, net, device, encoder, num_simulations=simulations, temp=current_temp)
        else:
            move = play_direct_policy(board, net, device, encoder, temp=current_temp)
            
        if move is None:
            break
            
        board.push(move)
        move_count += 1
        
    # Determine the game result
    if board.is_game_over():
        result = board.result()
        outcome = board.outcome()
        termination = outcome.termination.name if outcome else "UNKNOWN"
    else:
        result = "1/2-1/2"
        termination = "MOVE_LIMIT_REACHED"
        
    return white_name, black_name, result, termination, move_count


# ─── Model Discovery & Loader ────────────────────────────────────────────────

def detect_arch(state_dict):
    return 'se' if any('se.' in k for k in state_dict.keys()) else 'v1'


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    sd = ckpt['model_state_dict']
    arch = detect_arch(sd)
    
    if arch == 'se':
        net = AlphaZeroNet(num_blocks=12, num_channels=192)
        arch_label = 'SE-ResNet 12x192'
    else:
        net = AlphaZeroNetV1(num_blocks=10, num_channels=192)
        arch_label = 'Plain ResNet 10x192'
        
    net.load_state_dict(sd)
    net.to(device).eval()
    return net, arch_label


# ─── Elo Rating & Bradley-Terry Solver ───────────────────────────────────────

def compute_ratings_bt(games, players, iterations=200):
    """
    Computes Bradley-Terry maximum likelihood ratings for all players.
    Draws are handled as 0.5 wins for both.
    """
    p = {player: 1.0 for player in players}
    wins = {player: 0.0 for player in players}
    opponents = {player: [] for player in players}
    
    for p1, p2, outcome in games:
        # outcome is score for p1 (1.0 win, 0.5 draw, 0.0 loss)
        wins[p1] += outcome
        wins[p2] += (1.0 - outcome)
        opponents[p1].append(p2)
        opponents[p2].append(p1)
        
    # Run Minorization-Maximization algorithm for BT model
    for _ in range(iterations):
        new_p = {}
        for player in players:
            denom = 0.0
            for opp in opponents[player]:
                denom += 1.0 / (p[player] + p[opp])
            if denom > 0:
                new_p[player] = wins[player] / denom
            else:
                new_p[player] = p[player]
                
        # Normalize to prevent overflow/underflow
        sum_p = sum(new_p.values())
        if sum_p > 0:
            p = {player: val / sum_p for player, val in new_p.items()}
            
    # Convert to standard Elo scale (base 10, offset 1500)
    ratings = {}
    for player in players:
        if p[player] > 0:
            ratings[player] = 400.0 * math.log10(p[player])
        else:
            ratings[player] = -1000.0
            
    # Shift so that the average Elo is exactly 1500
    avg_rating = sum(ratings.values()) / len(players)
    for player in players:
        ratings[player] = ratings[player] - avg_rating + 1500.0
        
    return ratings


# ─── Main Program ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run simultaneous tournament evaluation of Chess AlphaZero models")
    parser.add_argument("--games-per-pair", type=int, default=2,
                        help="Number of games per pair (half as White, half as Black, must be even)")
    parser.add_argument("--simulations", type=int, default=50,
                        help="Number of MCTS simulations per move")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Maximum concurrent games running simultaneously")
    parser.add_argument("--no-mcts", action="store_true",
                        help="Use direct policy network predictions without MCTS (ultra-fast)")
    parser.add_argument("--max-moves", type=int, default=150,
                        help="Max moves per game before a draw is declared")
    parser.add_argument("--temp-moves", type=int, default=6,
                        help="Number of opening plies where temperature sampling is active")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device ('cuda' or 'cpu'). Auto-detected if not set.")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints",
                        help="Directory where model checkpoints are stored")
    
    args = parser.parse_args()
    
    # 1. Device Setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    logger.info("=========================================================")
    logger.info("♛ CHESS ALPHAZERO SIMULTANEOUS EVALUATION TOURNAMENT ♛")
    logger.info("=========================================================")
    logger.info(f"Start Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Compute Device   : {device.type.upper()}")
    if device.type == "cuda":
        logger.info(f"GPU Name         : {torch.cuda.get_device_name(0)}")
        
    # 2. Find and Load Models
    ckpt_dir = args.checkpoints_dir
    if not os.path.exists(ckpt_dir):
        logger.error(f"Checkpoint directory '{ckpt_dir}' does not exist. Exiting.")
        sys.exit(1)
        
    pth_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth") and "latest" not in f]
    # Include rl_latest if it exists
    if os.path.exists(os.path.join(ckpt_dir, "rl_latest.pth")):
        pth_files.append("rl_latest.pth")
        
    # De-duplicate
    pth_files = sorted(list(set(pth_files)))
    
    if len(pth_files) < 2:
        logger.error(f"Need at least 2 models in '{ckpt_dir}' to run a tournament. Found: {pth_files}")
        logger.error("Please ensure you have checkpoints like best_model.pth, model_ep5.pth etc.")
        sys.exit(1)
        
    logger.info(f"Found {len(pth_files)} model checkpoints. Loading...")
    
    models = {}
    encoder = Encoder()
    
    for fname in pth_files:
        name = fname.replace(".pth", "")
        path = os.path.join(ckpt_dir, fname)
        try:
            net, arch = load_model(path, device)
            models[name] = {
                "net": net,
                "arch": arch,
                "filename": fname
            }
            logger.info(f"  [✓] Loaded '{name}' ({arch})")
        except Exception as e:
            logger.error(f"  [✗] Failed to load checkpoint '{fname}': {e}")
            
    if len(models) < 2:
        logger.error("Failed to load enough models for a tournament. Exiting.")
        sys.exit(1)
        
    logger.info(f"Successfully loaded {len(models)} models into memory.")
    if device.type == "cuda":
        vram_alloc = torch.cuda.memory_allocated() / (1024**2)
        vram_res = torch.cuda.memory_reserved() / (1024**2)
        logger.info(f"GPU VRAM Usage   : {vram_alloc:.1f} MB allocated, {vram_res:.1f} MB reserved")
        
    # 3. Create Matchmaking Queue
    model_names = list(models.keys())
    match_list = []
    
    # Each unique pair playing White and Black games
    games_per_pair = args.games_per_pair
    # Make sure games_per_pair is even
    if games_per_pair % 2 != 0:
        games_per_pair += 1
        
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1 = model_names[i]
            m2 = model_names[j]
            # Half games m1 is White, half games m2 is White
            half = games_per_pair // 2
            for _ in range(half):
                match_list.append((m1, m2)) # m1=White, m2=Black
                match_list.append((m2, m1)) # m2=White, m1=Black
                
    # Shuffle matches to blend GPU workloads and prevent pattern bias
    random.shuffle(match_list)
    total_games = len(match_list)
    
    logger.info(f"Tournament Format:")
    logger.info(f"  Mode           : {'Direct Policy' if args.no_mcts else 'MCTS Search'}")
    if not args.no_mcts:
        logger.info(f"  Simulations    : {args.simulations} per move")
    logger.info(f"  Games Per Pair : {games_per_pair} ({games_per_pair//2} White / {games_per_pair//2} Black)")
    logger.info(f"  Temp Moves Plies: {args.temp_moves} (Randomness at start)")
    logger.info(f"  Concurrency    : {args.concurrency} simultaneous games")
    logger.info(f"  Total Games    : {total_games} matches")
    logger.info("=========================================================")
    logger.info("  Starting Matches...")
    logger.info("=========================================================")
    
    results = []
    # Statistics trackers
    wins = {name: 0 for name in model_names}
    losses = {name: 0 for name in model_names}
    draws = {name: 0 for name in model_names}
    total_moves = 0
    completed_games = 0
    
    # Head to Head matrix tracker
    # h2h[m1][m2] = [wins, draws, losses] from perspective of m1
    h2h = {m1: {m2: [0, 0, 0] for m2 in model_names} for m1 in model_names}
    
    start_tournament_time = time.time()
    
    # Run games concurrently
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for match_idx, (w_name, b_name) in enumerate(match_list):
            w_net = models[w_name]["net"]
            b_net = models[b_name]["net"]
            
            # Submit to thread pool
            future = executor.submit(
                play_game,
                w_name, w_net,
                b_name, b_net,
                device, encoder,
                use_mcts=not args.no_mcts,
                simulations=args.simulations,
                max_moves=args.max_moves,
                temp_moves=args.temp_moves
            )
            futures.append((future, match_idx))
            
        for future, idx in futures:
            try:
                w_name, b_name, result, term, moves = future.result()
                completed_games += 1
                total_moves += moves
                
                # Record result
                results.append((w_name, b_name, result))
                
                # Update tally
                if result == "1-0":
                    wins[w_name] += 1
                    losses[b_name] += 1
                    h2h[w_name][b_name][0] += 1  # win
                    h2h[b_name][w_name][2] += 1  # loss
                    outcome_str = f"{w_name} won as White"
                elif result == "0-1":
                    wins[b_name] += 1
                    losses[w_name] += 1
                    h2h[b_name][w_name][0] += 1  # win
                    h2h[w_name][b_name][2] += 1  # loss
                    outcome_str = f"{b_name} won as Black"
                else:
                    draws[w_name] += 1
                    draws[b_name] += 1
                    h2h[w_name][b_name][1] += 1  # draw
                    h2h[b_name][w_name][1] += 1  # draw
                    outcome_str = "Draw"
                    
                pct = (completed_games / total_games) * 100
                logger.info(
                    f"Match {completed_games:3d}/{total_games:3d} ({pct:5.1f}%): "
                    f"{w_name} (W) vs {b_name} (B) -> {result} ({outcome_str} in {moves} moves | Term: {term})"
                )
            except Exception as e:
                logger.error(f"Error executing match {idx}: {e}")
                
    elapsed_time = time.time() - start_tournament_time
    logger.info("=========================================================")
    logger.info("♛ TOURNAMENT COMPLETED ♛")
    logger.info(f"Total Time Elapsed: {elapsed_time:.1f} seconds ({elapsed_time/60:.2f} minutes)")
    logger.info(f"Avg Game Duration : {elapsed_time/total_games:.2f} seconds")
    logger.info(f"Avg Moves per Game: {total_moves/total_games:.1f} moves")
    logger.info("=========================================================")
    
    # 4. Solve Bradley-Terry Ratings (Elo Scale)
    # Convert results list to numerical outcomes (from perspective of m1)
    numerical_games = []
    for w_name, b_name, result in results:
        if result == "1-0":
            numerical_games.append((w_name, b_name, 1.0))
        elif result == "0-1":
            numerical_games.append((w_name, b_name, 0.0))
        else:
            numerical_games.append((w_name, b_name, 0.5))
            
    elo_ratings = compute_ratings_bt(numerical_games, model_names)
    
    # Sort leaderboard by rating
    leaderboard = sorted(model_names, key=lambda x: elo_ratings[x], reverse=True)
    
    # 5. Generate beautiful leaderboard & logs
    logger.info("RANKING LEADERBOARD (Bradley-Terry Elo equivalent):")
    logger.info(f"{'Rank':<5} | {'Model Name':<25} | {'Elo':<6} | {'W / D / L':<12} | {'Points':<6} | {'Win Rate':<8}")
    logger.info("-" * 75)
    
    for rank, name in enumerate(leaderboard, 1):
        w = wins[name]
        d = draws[name]
        l = losses[name]
        pts = w + 0.5 * d
        tot = w + d + l
        wr = (pts / tot) * 100 if tot > 0 else 0
        elo_str = f"{round(elo_ratings[name])}"
        logger.info(f"{rank:<5} | {name:<25} | {elo_str:<6} | {w:2d}/{d:2d}/{l:2d}   | {pts:<6.1f} | {wr:<7.1f}%")
        
    logger.info("=========================================================")
    
    # 6. Print Head-to-Head Detailed Grid
    logger.info("HEAD-TO-HEAD MATRIX (Row vs Column: Wins / Draws / Losses):")
    
    # Header row
    hdr = f"{'Model Name':<25} |"
    for name in model_names:
        hdr += f" {name[:12]:<12} |"
    logger.info(hdr)
    logger.info("-" * len(hdr))
    
    for row_name in model_names:
        row_str = f"{row_name:<25} |"
        for col_name in model_names:
            if row_name == col_name:
                row_str += f" {'-':^12} |"
            else:
                stats = h2h[row_name][col_name]
                stats_str = f"{stats[0]}/{stats[1]}/{stats[2]}"
                row_str += f" {stats_str:^12} |"
        logger.info(row_str)
        
    logger.info("=========================================================")
    logger.info(f"Detailed logs saved to: {LOG_FILE}")
    logger.info("=========================================================")


if __name__ == "__main__":
    main()
