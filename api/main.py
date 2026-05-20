"""
Chess AI REST API Service
Fast, lightweight, developer-friendly backend v1.0
Runs on port 3210.
"""

import sys
import os
import re
import math
import random
import asyncio
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Ensure parent directory is in sys.path to import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import chess
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Local imports
from model.alphazero_net_v1 import AlphaZeroNetV1
from model.alphazero_net import AlphaZeroNet
from env.encoder import Encoder, MOVE_TO_INDEX, NUM_MOVES
from web.mcts_inference import search_mcts

# ─── Concurrency Configuration ───────────────────────────────────────────────
# Limit concurrent GPU/CPU intensive tasks (MCTS & Inference) to prevent OOM/overload
# and keep the API server highly responsive.
MAX_CONCURRENT_TASKS = int(os.environ.get("CHESS_CONCURRENT_TASKS", 2))
chess_executor = ThreadPoolExecutor(
    max_workers=max(4, MAX_CONCURRENT_TASKS * 2),
    thread_name_prefix="chess_ai_worker"
)
gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

# ─── Device Detection ────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[API] Initializing Chess AI service...")
print(f"[API] Compute device detected: {device.type.upper()}")

# Path to checkpoints
CKPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))

# ─── Helpers ─────────────────────────────────────────────────────────────────
def _detect_arch(state_dict: dict) -> str:
    """Return 'se' (AlphaZeroNet SE-ResNet 12×192) or 'v1' (AlphaZeroNetV1 Plain 10×192)."""
    return 'se' if any('se.' in k for k in state_dict.keys()) else 'v1'

def _load_model(path: str):
    ckpt = torch.load(path, map_location=device)
    sd   = ckpt['model_state_dict']
    arch = _detect_arch(sd)
    if arch == 'se':
        net = AlphaZeroNet(num_blocks=12, num_channels=192)
        arch_label = 'SE-ResNet 12×192'
    else:
        net = AlphaZeroNetV1(num_blocks=10, num_channels=192)
        arch_label = 'Plain ResNet 10×192'
    net.load_state_dict(sd)
    net.to(device).eval()
    return net, ckpt, arch, arch_label

# Display name mappings
_DISPLAY = {
    'best_model':          'V1 Best  (Plain ResNet)',
    'model_ep1_batch2000': 'Pretrain EP1-B2000',
    'model_ep5':           'SE-ResNet EP5'
}

# ─── Load all checkpoints on startup ─────────────────────────────────────────
MODELS: Dict[str, dict] = {}
DEFAULT_MODEL = "best_model"

if os.path.exists(CKPT_DIR):
    pth_files = sorted(f for f in os.listdir(CKPT_DIR) if f.endswith('.pth'))
    for fname in pth_files:
        key = fname.replace('.pth', '')
        path = os.path.join(CKPT_DIR, fname)
        try:
            net, ckpt, arch, arch_label = _load_model(path)
            display = _DISPLAY.get(key, key)
            if "rl" in key:
                category = "Reinforcement Learning"
            elif "v1_finetuned" in key:
                category = "V1 True Finetuning (6M Elite)"
            elif arch == 'se':
                category = "SE-ResNet (Failed Transfer)"
            else:
                category = "AlphaZero V1 (Pretrained 20M)"

            MODELS[key] = dict(
                net=net, ckpt=ckpt, arch=arch, category=category,
                arch_label=arch_label, display=display, filename=fname
            )
            ep   = ckpt.get('epoch', '?')
            loss = ckpt.get('loss',  0.0)
            if isinstance(loss, float) and math.isnan(loss):
                loss = 0.0
            print(f"[API] Loaded model '{key}' (arch: {arch_label}, epoch: {ep}, loss: {loss:.4f})")
        except Exception as e:
            print(f"[API] ERROR loading checkpoint '{fname}': {e}")
else:
    print(f"[API] WARNING: Checkpoint directory '{CKPT_DIR}' not found!")

# Fallback default model detection
if MODELS:
    for candidate in ['rl_latest_4', 'rl_latest_3', 'rl_latest', 'best_model']:
        if candidate in MODELS:
            DEFAULT_MODEL = candidate
            break
    if DEFAULT_MODEL not in MODELS:
        DEFAULT_MODEL = next(iter(MODELS))
    print(f"[API] Default active model: '{DEFAULT_MODEL}' (Total loaded: {len(MODELS)})")
else:
    print("[API] WARNING: No models loaded. Running with random/heuristic fallback only!")
    DEFAULT_MODEL = None

encoder = Encoder()

# ─── FastAPI Application & CORS ──────────────────────────────────────────────
app = FastAPI(
    title="Chess AI Backend API Service",
    description="Stateless REST API for Chess AI player move submission and board state analysis.",
    version="1.0.0"
)

# Enable CORS for local applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for seamless local access (localhost, mobile apps, desktop)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic Request & Response Schemas ─────────────────────────────────────
class ModelInfo(BaseModel):
    key: str
    display: str
    arch: str
    arch_type: str
    category: str
    filename: str
    epoch: int
    loss: float

class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    default: Optional[str]

class StatusResponse(BaseModel):
    status: str
    device: str
    total_models: int
    default_model: Optional[str]
    cuda_available: bool
    max_concurrent_tasks: int

class MoveCandidate(BaseModel):
    uci: str
    san: str
    prob: float

class ChessMoveRequest(BaseModel):
    fen: str = Field(default=chess.STARTING_FEN, description="Board position in FEN notation.")
    move_history: Optional[List[str]] = Field(default=None, description="Optional history of SAN/UCI moves played.")
    player_move: Optional[str] = Field(default=None, description="Optional move from the player to push prior to AI turn (SAN or UCI).")
    model: Optional[str] = Field(default=None, description="Model key to use for evaluation. Falls back to default if not found.")
    difficulty: Optional[int] = Field(default=3, ge=1, le=4, description="Difficulty levels: 1 (Random), 2 (Medium-Softmax), 3 (Hard-Direct), 4 (Expert-MCTS).")

class ChessMoveResponse(BaseModel):
    player_move: Optional[str] = None
    player_move_san: Optional[str] = None
    ai_move: Optional[str] = None
    ai_move_san: Optional[str] = None
    fen: str
    is_game_over: bool
    result: Optional[str] = None
    termination: Optional[str] = None
    evaluation: float
    top_moves: List[MoveCandidate] = []

class AIFirstRequest(BaseModel):
    fen: str = Field(default=chess.STARTING_FEN, description="Board position in FEN notation.")
    model: Optional[str] = Field(default=None, description="Model key to use.")
    difficulty: Optional[int] = Field(default=3, ge=1, le=4, description="Difficulty levels: 1 to 4.")

class LegalMovesRequest(BaseModel):
    fen: str = Field(default=chess.STARTING_FEN)

class LegalMovesResponse(BaseModel):
    fen: str
    legal_moves: List[str]
    legal_moves_san: List[str]

# ─── Robust Move Helper ──────────────────────────────────────────────────────
def push_move_robust(board: chess.Board, move_str: str) -> chess.Move:
    """
    Tries to parse and push a move in either UCI or SAN format.
    Raises ValueError if the move is invalid or not legal.
    """
    move_str = move_str.strip()
    # Try parsing as UCI
    try:
        move = chess.Move.from_uci(move_str)
        if move in board.legal_moves:
            board.push(move)
            return move
    except Exception:
        pass

    # Try parsing as SAN
    try:
        move = board.push_san(move_str)
        return move
    except Exception:
        pass

    raise ValueError(f"Move '{move_str}' is invalid or illegal in this position.")

# ─── Core AI Move Solver ─────────────────────────────────────────────────────
def get_ai_move(board: chess.Board, difficulty: int = 3, model_key: str = None) -> tuple[Optional[str], float, list]:
    legal = list(board.legal_moves)
    if not legal:
        return None, 0.0, []

    # Difficulty 1: pure random
    if difficulty == 1:
        chosen = random.choice(legal)
        # Evaluate randomly selected position simply
        board.push(chosen)
        val = 0.0
        board.pop()
        return chosen.uci(), val, [{"uci": chosen.uci(), "san": board.san(chosen), "prob": 1.0}]

    # Retrieve network
    model_key = model_key or DEFAULT_MODEL
    entry = MODELS.get(model_key) or (MODELS.get(DEFAULT_MODEL) if DEFAULT_MODEL else None)
    
    if not entry:
        # Fallback if no model is loaded
        chosen = random.choice(legal)
        return chosen.uci(), 0.0, [{"uci": chosen.uci(), "san": board.san(chosen), "prob": 1.0}]

    net = entry['net']

    # Expert level (Difficulty 4) or Reinforcement Learning models
    # use MCTS search for stronger play
    if difficulty == 4 or 'rl' in str(model_key).lower():
        num_sims = 800 if difficulty == 4 else 300
        # MCTS returns the best uci move, root value, and top moves with probabilities
        uci, val, top_moves_list = search_mcts(board, net, device, num_simulations=num_sims)
        
        # Enrich MCTS top moves with SAN strings
        enriched_top_moves = []
        for cand in top_moves_list:
            try:
                m = chess.Move.from_uci(cand["uci"])
                san_str = board.san(m)
                enriched_top_moves.append({
                    "uci": cand["uci"],
                    "san": san_str,
                    "prob": cand["prob"]
                })
            except Exception:
                enriched_top_moves.append({
                    "uci": cand["uci"],
                    "san": cand["uci"],
                    "prob": cand["prob"]
                })
        return uci, val, enriched_top_moves

    # Direct policy network logic for Difficulty 2 & 3 (Fast, no MCTS simulations)
    tensor_t = torch.from_numpy(
        encoder.board_to_tensor(board)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, value = net(tensor_t)

    value_score = value.item()
    logits      = policy_logits[0].cpu().numpy()

    legal_indices = []
    legal_moves_map = {}
    for mv in legal:
        idx = MOVE_TO_INDEX.get(mv.uci())
        if idx is not None:
            legal_indices.append(idx)
            legal_moves_map[idx] = mv.uci()

    if not legal_indices:
        # Fallback to first legal move
        chosen = legal[0]
        return chosen.uci(), value_score, [{"uci": chosen.uci(), "san": board.san(chosen), "prob": 1.0}]

    masked = np.full(NUM_MOVES, -1e9, dtype=np.float32)
    for idx in legal_indices:
        masked[idx] = logits[idx]

    # Level 2 uses high temperature (1.5) for high exploration, Level 3 uses low temperature (0.5) for direct play
    temp = 1.5 if difficulty == 2 else 0.5
    ll = masked[legal_indices] / temp
    ll -= ll.max()
    probs = np.exp(ll)
    probs /= probs.sum()

    sorted_idx = np.argsort(probs)[::-1]
    top_moves = []
    for r in sorted_idx[:5]:
        uci_str = legal_moves_map[legal_indices[r]]
        try:
            m = chess.Move.from_uci(uci_str)
            san_str = board.san(m)
        except Exception:
            san_str = uci_str
        top_moves.append({
            "uci": uci_str,
            "san": san_str,
            "prob": float(probs[r])
        })

    # Pick move based on probability distribution for Level 2, or pure argmax for Level 3
    if difficulty == 2:
        chosen_idx = np.random.choice(len(legal_indices), p=probs)
    else:
        chosen_idx = int(np.argmax(probs))

    chosen_uci = legal_moves_map[legal_indices[chosen_idx]]
    return chosen_uci, value_score, top_moves


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get general API health status and loaded model counts."""
    return StatusResponse(
        status="online",
        device=str(device),
        total_models=len(MODELS),
        default_model=DEFAULT_MODEL,
        cuda_available=torch.cuda.is_available(),
        max_concurrent_tasks=MAX_CONCURRENT_TASKS
    )


@app.get("/api/models", response_model=ModelsListResponse)
async def list_models():
    """List details of all loaded models in checkpoints/."""
    result = []
    for key, m in MODELS.items():
        ckpt = m['ckpt']
        loss = ckpt.get('loss', 0.0)
        if isinstance(loss, float) and math.isnan(loss):
            loss = 0.0
            
        result.append(ModelInfo(
            key=key,
            display=m['display'],
            arch=m['arch_label'],
            arch_type=m['arch'],
            category=m['category'],
            filename=m['filename'],
            epoch=int(ckpt.get('epoch', 0)),
            loss=float(loss)
        ))
    return ModelsListResponse(models=result, default=DEFAULT_MODEL)


@app.post("/api/move", response_model=ChessMoveResponse)
async def make_move(req: ChessMoveRequest = Body(...)):
    """
    Perform a turn. Pushes the player's last move (if provided) onto the board,
    checks for terminal status, then uses the specified model to make an AI move.
    """
    try:
        # Initialize board
        if req.move_history:
            board = chess.Board()
            for m in req.move_history:
                push_move_robust(board, m)
        else:
            board = chess.Board(req.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid game history or FEN: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse FEN/History: {str(e)}")

    # 1. Handle player's move if provided
    player_move_uci = None
    player_move_san = None

    if req.player_move:
        if board.is_game_over():
            raise HTTPException(status_code=400, detail="Cannot apply player move; the game is already over.")
        try:
            # First, parse the move to a valid chess.Move object BEFORE pushing it.
            # This allows us to generate both SAN and UCI formats correctly in the current state.
            move = None
            
            # Try parsing as UCI
            try:
                move = chess.Move.from_uci(req.player_move)
                if move not in board.legal_moves:
                    move = None
            except Exception:
                pass
                
            # If not UCI, try parsing as SAN
            if move is None:
                try:
                    move = board.parse_san(req.player_move)
                except Exception:
                    pass
            
            if move is None or move not in board.legal_moves:
                raise ValueError(f"Move '{req.player_move}' is invalid or illegal in this position.")
                
            # Record SAN and UCI formats in the state prior to execution
            player_move_san = board.san(move)
            player_move_uci = move.uci()
            
            # Push the move onto the board
            board.push(move)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 2. Check if board is game over AFTER player's move
    if board.is_game_over():
        term = board.outcome().termination.name if board.outcome() else "UNKNOWN"
        return ChessMoveResponse(
            player_move=player_move_uci,
            player_move_san=player_move_san,
            ai_move=None,
            ai_move_san=None,
            fen=board.fen(),
            is_game_over=True,
            result=board.result(),
            termination=term,
            evaluation=0.0,
            top_moves=[]
        )

    # 3. Request AI move
    difficulty = req.difficulty if req.difficulty is not None else 3
    model_key = req.model if req.model in MODELS else DEFAULT_MODEL

    try:
        # Offload heavy CPU/GPU computation to the background thread pool
        # and limit concurrency to prevent VRAM exhaustion / system thrashing.
        async with gpu_semaphore:
            loop = asyncio.get_running_loop()
            ai_uci, evaluation, top_moves_list = await loop.run_in_executor(
                chess_executor,
                get_ai_move,
                board,
                difficulty,
                model_key
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Move generation error: {str(e)}")

    if ai_uci is None:
        raise HTTPException(status_code=400, detail="No legal moves available for the AI.")

    # Apply AI move
    ai_move = chess.Move.from_uci(ai_uci)
    ai_san = board.san(ai_move)
    board.push(ai_move)

    # Check terminal state again
    is_over = board.is_game_over()
    result_str = board.result() if is_over else None
    termination_str = board.outcome().termination.name if is_over and board.outcome() else None

    # Cast raw dict list into Pydantic models
    candidates = [
        MoveCandidate(uci=m["uci"], san=m["san"], prob=m["prob"])
        for m in top_moves_list
    ]

    return ChessMoveResponse(
        player_move=player_move_uci,
        player_move_san=player_move_san,
        ai_move=ai_uci,
        ai_move_san=ai_san,
        fen=board.fen(),
        is_game_over=is_over,
        result=result_str,
        termination=termination_str,
        evaluation=evaluation,
        top_moves=candidates
    )


@app.post("/api/ai-move-first", response_model=ChessMoveResponse)
async def ai_move_first(req: AIFirstRequest = Body(...)):
    """
    Triggered when AI starts the game or makes the first move from a given position.
    No player move is expected in the request.
    """
    try:
        board = chess.Board(req.fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN position: {str(e)}")

    if board.is_game_over():
        term = board.outcome().termination.name if board.outcome() else "UNKNOWN"
        return ChessMoveResponse(
            fen=board.fen(),
            is_game_over=True,
            result=board.result(),
            termination=term,
            evaluation=0.0
        )

    difficulty = req.difficulty if req.difficulty is not None else 3
    model_key = req.model if req.model in MODELS else DEFAULT_MODEL

    try:
        # Offload heavy CPU/GPU computation to the background thread pool
        # and limit concurrency to prevent VRAM exhaustion / system thrashing.
        async with gpu_semaphore:
            loop = asyncio.get_running_loop()
            ai_uci, evaluation, top_moves_list = await loop.run_in_executor(
                chess_executor,
                get_ai_move,
                board,
                difficulty,
                model_key
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Move generation error: {str(e)}")

    if ai_uci is None:
        raise HTTPException(status_code=400, detail="No legal moves available for the AI.")

    ai_move = chess.Move.from_uci(ai_uci)
    ai_san = board.san(ai_move)
    board.push(ai_move)

    is_over = board.is_game_over()
    result_str = board.result() if is_over else None
    termination_str = board.outcome().termination.name if is_over and board.outcome() else None

    candidates = [
        MoveCandidate(uci=m["uci"], san=m["san"], prob=m["prob"])
        for m in top_moves_list
    ]

    return ChessMoveResponse(
        ai_move=ai_uci,
        ai_move_san=ai_san,
        fen=board.fen(),
        is_game_over=is_over,
        result=result_str,
        termination=termination_str,
        evaluation=evaluation,
        top_moves=candidates
    )


@app.post("/api/legal_moves", response_model=LegalMovesResponse)
async def get_legal_moves(req: LegalMovesRequest = Body(...)):
    """Fetch legal moves for the given FEN board state."""
    try:
        board = chess.Board(req.fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {str(e)}")

    moves_uci = [m.uci() for m in board.legal_moves]
    moves_san = [board.san(m) for m in board.legal_moves]

    return LegalMovesResponse(
        fen=board.fen(),
        legal_moves=moves_uci,
        legal_moves_san=moves_san
    )
