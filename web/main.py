"""
Chess AI Web Server — FastAPI backend v3
Auto-discovers all *.pth in checkpoints/, detects arch (Plain ResNet v1 vs SE-ResNet),
loads every model and exposes them to the frontend.
"""
import sys, os, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import chess
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import random

from model.alphazero_net_v1 import AlphaZeroNetV1
from model.alphazero_net   import AlphaZeroNet
from env.encoder import Encoder, MOVE_TO_INDEX, NUM_MOVES

# ─── Device ──────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[AI] Device: {device}")

CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')

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

# ─── Pretty display names ────────────────────────────────────────────────────
_DISPLAY = {
    'best_model':          'V1 Best  (Plain ResNet)',
    'finetuned_best':      'Finetuned Best  (SE-ResNet ep4)',
    'model_ep1_batch2000': 'EP1-B2000  (V1 Plain)',
    'model_ep1_batch4000': 'EP1-B4000  (V1 Plain)',
    'model_ep1':           'EP1  (SE-ResNet)',
    'model_ep2_batch2000': 'EP2-B2000  (V1 Plain)',
    'model_ep2':           'EP2  (SE-ResNet)',
    'model_ep3':           'EP3  (SE-ResNet)',
    'model_ep4':           'EP4  (SE-ResNet)',
    'model_ep5':           'EP5  (SE-ResNet)',
}

# ─── Load all checkpoints ─────────────────────────────────────────────────────
MODELS: dict[str, dict] = {}   # key → {net, ckpt, arch, arch_label, display, filename}

pth_files = sorted(f for f in os.listdir(CKPT_DIR) if f.endswith('.pth'))
for fname in pth_files:
    key = fname.replace('.pth', '')
    path = os.path.join(CKPT_DIR, fname)
    try:
        net, ckpt, arch, arch_label = _load_model(path)
        display = _DISPLAY.get(key, key)
        MODELS[key] = dict(net=net, ckpt=ckpt, arch=arch,
                           arch_label=arch_label, display=display, filename=fname)
        ep   = ckpt.get('epoch', '?')
        loss = ckpt.get('loss',  float('nan'))
        print(f"[AI] Loaded '{key}'  arch={arch_label}  ep={ep}  loss={loss:.4f}")
    except Exception as e:
        print(f"[AI] SKIP '{fname}': {e}")

# Fallback: make sure there's always a default
DEFAULT_MODEL = 'best_model' if 'best_model' in MODELS else next(iter(MODELS))
print(f"[AI] Default model: {DEFAULT_MODEL}  |  Total loaded: {len(MODELS)}")

encoder = Encoder()

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Chess AI", version="3.0")

static_dir = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ─── Schemas ─────────────────────────────────────────────────────────────────
class MoveRequest(BaseModel):
    fen:        str
    difficulty: Optional[int] = 3
    model:      Optional[str] = DEFAULT_MODEL


class MoveResponse(BaseModel):
    move:         str
    fen:          str
    is_game_over: bool
    result:       Optional[str]
    evaluation:   float
    top_moves:    list


# ─── AI Logic ────────────────────────────────────────────────────────────────
def get_ai_move(board: chess.Board, difficulty: int = 3,
                model_key: str = DEFAULT_MODEL) -> tuple[str, float, list]:
    legal = list(board.legal_moves)
    if not legal:
        return None, 0.0, []

    if difficulty == 1:
        return random.choice(legal).uci(), 0.0, []

    entry = MODELS.get(model_key) or MODELS[DEFAULT_MODEL]
    net   = entry['net']

    tensor_t = torch.from_numpy(
        encoder.board_to_tensor(board)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, value = net(tensor_t)

    value_score = value.item()
    logits      = policy_logits[0].cpu().numpy()

    legal_indices   = []
    legal_moves_map = {}
    for mv in legal:
        idx = MOVE_TO_INDEX.get(mv.uci())
        if idx is not None:
            legal_indices.append(idx)
            legal_moves_map[idx] = mv.uci()

    if not legal_indices:
        return legal[0].uci(), value_score, []

    masked = np.full(NUM_MOVES, -1e9, dtype=np.float32)
    for idx in legal_indices:
        masked[idx] = logits[idx]

    temp = {2: 1.5, 3: 0.5, 4: 0.1}.get(difficulty, 0.5)
    ll   = masked[legal_indices] / temp
    ll  -= ll.max()
    probs = np.exp(ll); probs /= probs.sum()

    sorted_idx = np.argsort(probs)[::-1]
    top_moves  = [{"uci": legal_moves_map[legal_indices[r]],
                   "prob": float(probs[r])} for r in sorted_idx[:5]]

    chosen = (np.random.choice(len(legal_indices), p=probs)
              if difficulty <= 2 else int(np.argmax(probs)))
    return legal_moves_map[legal_indices[chosen]], value_score, top_moves


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(static_dir, 'index.html'), encoding='utf-8') as f:
        return f.read()


@app.post("/api/move", response_model=MoveResponse)
async def make_ai_move(req: MoveRequest):
    try:
        board = chess.Board(req.fen)
    except Exception:
        raise HTTPException(400, "FEN không hợp lệ")

    if board.is_game_over():
        return MoveResponse(move="", fen=board.fen(), is_game_over=True,
                            result=board.result(), evaluation=0.0, top_moves=[])

    key = req.model if req.model in MODELS else DEFAULT_MODEL
    uci, eval_score, top_moves = get_ai_move(board, req.difficulty, key)
    if uci is None:
        raise HTTPException(400, "Không có nước đi hợp lệ")

    board.push(chess.Move.from_uci(uci))
    return MoveResponse(move=uci, fen=board.fen(),
                        is_game_over=board.is_game_over(),
                        result=board.result() if board.is_game_over() else None,
                        evaluation=eval_score, top_moves=top_moves)


@app.get("/api/models")
async def list_models():
    """Returns all loaded models with metadata — used by frontend to build the selector."""
    result = []
    for key, m in MODELS.items():
        ckpt = m['ckpt']
        result.append({
            "key":        key,
            "display":    m['display'],
            "arch":       m['arch_label'],
            "arch_type":  m['arch'],          # 'v1' | 'se'
            "filename":   m['filename'],
            "epoch":      int(ckpt.get('epoch', 0)),
            "loss":       float(ckpt.get('loss', float('nan'))),
        })
    return {"models": result, "default": DEFAULT_MODEL}


@app.get("/api/status")
async def status():
    model_info = {k: f"{m['arch_label']} — ep {m['ckpt'].get('epoch',0)}, loss {m['ckpt'].get('loss',0):.4f}"
                  for k, m in MODELS.items()}
    return {
        "status":  "online",
        "device":  str(device),
        "models":  model_info,
        "total_models": len(MODELS),
        # legacy compat
        "model":            "Multi-model Chess AI",
        "checkpoint_epoch": 0,
        "checkpoint_loss":  0.0,
        "num_moves":        NUM_MOVES,
    }


@app.post("/api/legal_moves")
async def get_legal_moves(body: dict):
    try:
        board = chess.Board(body.get("fen", chess.STARTING_FEN))
    except Exception:
        raise HTTPException(400, "FEN không hợp lệ")
    return {"legal_moves": [m.uci() for m in board.legal_moves]}
