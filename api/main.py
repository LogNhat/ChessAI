"""
Chess AI REST API Service
Fast, lightweight, developer-friendly backend v1.0.0
"""

import sys
import os
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

# ─── Concurrency & Compute Configurations ────────────────────────────────────
MAX_CONCURRENT_TASKS = int(os.environ.get("CHESS_CONCURRENT_TASKS", 2))
chess_executor = ThreadPoolExecutor(
    max_workers=max(4, MAX_CONCURRENT_TASKS * 2),
    thread_name_prefix="chess_ai_worker"
)
gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger_info = lambda msg: print(f"[API] {msg}")
logger_info(f"Compute device detected: {device.type.upper()}")

# ─── Checkpoint Loading ──────────────────────────────────────────────────────
CKPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
MODELS: Dict[str, dict] = {}
DEFAULT_MODEL: Optional[str] = None

def _detect_arch(state_dict: dict) -> str:
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

_ELO = {
    'model_ep1': 1206,
    'model_ep2': 1248,
    'model_ep1_batch2000': 1298,
    'model_ep3': 1373,
    'model_ep4': 1467,
    'model_ep2_batch2000': 1501,
    'model_ep1_batch4000': 1516,
    'finetuned_best': 1516,
    'best_model': 1518,
    'model_ep5': 1519,
    'v1_finetuned_ep3': 1586,
    'v1_finetuned_ep2': 1604,
    'v1_finetuned_ep1': 1608,
    'v1_finetuned_ep4': 1644,
    'v1_finetuned_best': 1659,
    'v1_finetuned_ep5': 1736,
}

_DISPLAY = {
    k: f"{k} (Elo {v})" for k, v in _ELO.items()
}

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
            loss = ckpt.get('loss', 0.0)
            if isinstance(loss, float) and math.isnan(loss):
                loss = 0.0
            logger_info(f"Loaded '{key}' ({arch_label}, epoch: {ep}, loss: {loss:.4f})")
        except Exception as e:
            logger_info(f"ERROR loading checkpoint '{fname}': {e}")
            
    # Resolve default model
    for candidate in ['rl_latest_4', 'rl_latest_3', 'rl_latest', 'best_model']:
        if candidate in MODELS:
            DEFAULT_MODEL = candidate
            break
    if not DEFAULT_MODEL and MODELS:
        DEFAULT_MODEL = next(iter(MODELS))
    logger_info(f"Default active model: '{DEFAULT_MODEL}'")
else:
    logger_info(f"WARNING: Checkpoint directory '{CKPT_DIR}' not found!")

encoder = Encoder()

# ─── FastAPI Configuration ───────────────────────────────────────────────────
app = FastAPI(
    title="Chess AI REST API Service",
    description="Stateless REST API for Chess AI player move submission and board state analysis.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic Schemas ────────────────────────────────────────────────────────
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
    player_move: Optional[str] = Field(default=None, description="Optional move from the player (SAN or UCI).")
    model: Optional[str] = Field(default=None, description="Model key to use for evaluation.")
    difficulty: Optional[int] = Field(default=3, ge=1, le=4, description="Difficulty levels: 1 (Direct Policy - No MCTS), 2 (MCTS 100 sims), 3 (MCTS 300 sims), 4 (MCTS 800 sims).")

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

# ─── Robust Move Push Helper ──────────────────────────────────────────────────
def push_move_robust(board: chess.Board, move_str: str) -> chess.Move:
    move_str = move_str.strip()
    try:
        move = chess.Move.from_uci(move_str)
        if move in board.legal_moves:
            board.push(move)
            return move
    except Exception:
        pass

    try:
        move = board.push_san(move_str)
        return move
    except Exception:
        pass

    raise ValueError(f"Move '{move_str}' is invalid or illegal.")

# ─── Core AI Move Solver ─────────────────────────────────────────────────────
def get_ai_move(board: chess.Board, difficulty: int = 3, model_key: str = None) -> tuple[Optional[str], float, list]:
    legal = list(board.legal_moves)
    if not legal:
        return None, 0.0, []

    # Resolve active model
    model_key = model_key or DEFAULT_MODEL
    entry = MODELS.get(model_key) or (MODELS.get(DEFAULT_MODEL) if DEFAULT_MODEL else None)
    
    if not entry:
        chosen = random.choice(legal)
        return chosen.uci(), 0.0, [{"uci": chosen.uci(), "san": board.san(chosen), "prob": 1.0}]

    net = entry['net']

    # Difficulty mappings:
    # 1: Direct Policy prediction (no MCTS)
    # 2: MCTS with 100 simulations
    # 3: MCTS with 300 simulations
    # 4: MCTS with 800 simulations
    if difficulty > 1:
        simulations_map = {2: 100, 3: 300, 4: 800}
        num_sims = simulations_map.get(difficulty, 300)
        
        uci, val, top_moves_list = search_mcts(board, net, device, num_simulations=num_sims)
        
        enriched_top_moves = []
        for cand in top_moves_list:
            try:
                m = chess.Move.from_uci(cand["uci"])
                san_str = board.san(m)
                enriched_top_moves.append({
                    "uci": cand["uci"], "san": san_str, "prob": cand["prob"]
                })
            except Exception:
                enriched_top_moves.append({
                    "uci": cand["uci"], "san": cand["uci"], "prob": cand["prob"]
                })
        return uci, val, enriched_top_moves

    # Difficulty == 1: Direct policy network predictions (no MCTS simulations)
    tensor_t = torch.from_numpy(encoder.board_to_tensor(board)).unsqueeze(0).to(device)
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
        chosen = legal[0]
        return chosen.uci(), value_score, [{"uci": chosen.uci(), "san": board.san(chosen), "prob": 1.0}]

    masked = np.full(NUM_MOVES, -1e9, dtype=np.float32)
    for idx in legal_indices:
        masked[idx] = logits[idx]

    # Standard temperature softmax for direct policy (using temp=0.5 for direct clean play)
    ll = masked[legal_indices] / 0.5
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
            "uci": uci_str, "san": san_str, "prob": float(probs[r])
        })

    chosen_idx = int(np.argmax(probs))
    chosen_uci = legal_moves_map[legal_indices[chosen_idx]]
    return chosen_uci, value_score, top_moves

# ─── FastAPI Endpoints ───────────────────────────────────────────────────────
@app.get("/api/status", response_model=StatusResponse)
async def get_status():
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
    # Sort models by Elo rating in increasing order
    result.sort(key=lambda x: _ELO.get(x.key, 1500))
    return ModelsListResponse(models=result, default=DEFAULT_MODEL)

@app.post("/api/move", response_model=ChessMoveResponse)
async def make_move(req: ChessMoveRequest = Body(...)):
    try:
        if req.move_history:
            board = chess.Board()
            for m in req.move_history:
                push_move_robust(board, m)
        else:
            board = chess.Board(req.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid game history or FEN: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse position: {str(e)}")

    player_move_uci = None
    player_move_san = None

    if req.player_move:
        if board.is_game_over():
            raise HTTPException(status_code=400, detail="Cannot push player move; game is already over.")
        try:
            move = None
            try:
                move = chess.Move.from_uci(req.player_move)
                if move not in board.legal_moves:
                    move = None
            except Exception:
                pass
                
            if move is None:
                try:
                    move = board.parse_san(req.player_move)
                except Exception:
                    pass
            
            if move is None or move not in board.legal_moves:
                raise ValueError(f"Move '{req.player_move}' is illegal.")
                
            player_move_san = board.san(move)
            player_move_uci = move.uci()
            board.push(move)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    if board.is_game_over():
        term = board.outcome().termination.name if board.outcome() else "UNKNOWN"
        return ChessMoveResponse(
            player_move=player_move_uci,
            player_move_san=player_move_san,
            fen=board.fen(),
            is_game_over=True,
            result=board.result(),
            termination=term,
            evaluation=0.0
        )

    difficulty = req.difficulty if req.difficulty is not None else 3
    model_key = req.model if req.model in MODELS else DEFAULT_MODEL

    try:
        async with gpu_semaphore:
            loop = asyncio.get_running_loop()
            ai_uci, evaluation, top_moves_list = await loop.run_in_executor(
                chess_executor, get_ai_move, board, difficulty, model_key
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Move generation error: {str(e)}")

    if ai_uci is None:
        raise HTTPException(status_code=400, detail="No legal moves available for AI.")

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
        async with gpu_semaphore:
            loop = asyncio.get_running_loop()
            ai_uci, evaluation, top_moves_list = await loop.run_in_executor(
                chess_executor, get_ai_move, board, difficulty, model_key
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Move generation error: {str(e)}")

    if ai_uci is None:
        raise HTTPException(status_code=400, detail="No legal moves available.")

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
