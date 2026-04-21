import math
import chess
import torch
import torch.nn.functional as F
import numpy as np
from env.encoder import Encoder, MOVE_TO_INDEX

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
        # Trách hòa tuyệt đối: Trả về 5.0 để tạo ra Q-value cực âm ở nút cha (-4.95). 
        # Thậm chí tệ hơn cả bị chiếu bí (v=-1.0). 
        # Bắt AI thà tự vứt quân mình, thua luôn, chứ không được phép hòa.
        return 5.0
    return -1.0 

def apply_endgame_heuristic(b, v):
    # Nếu còn <= 5 quân cờ và một bên đang thắng thế rõ ràng, MCTS hay bị phẳng gía trị dẫn tới đi lòng vòng
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
            # Ưu tiên dồn vua thua vào góc tường
            r_l = chess.square_rank(l_king)
            f_l = chess.square_file(l_king)
            edge = max(r_l, 7 - r_l) + max(f_l, 7 - f_l)
            
            # Ưu tiên vua thắng tiến lại gần vua thua
            r_w = chess.square_rank(w_king)
            f_w = chess.square_file(w_king)
            dist = max(abs(r_l - r_w), abs(f_l - f_w))
            
            h = (edge * 0.01) - (dist * 0.01)
            v += sign * h
            return max(-1.0, min(1.0, v))
    return v


def search_mcts(board, net, device, num_simulations=100):
    encoder = Encoder()
    root = MCTSNode()
    
    tensor_t = torch.from_numpy(encoder.board_to_tensor(board)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = net(tensor_t)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        root_value = apply_endgame_heuristic(board, value.item())
        
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
                v = 5.0 # Bị phạt 5.0 nếu lặp nước (thậm chí lặp 2 lần) hoặc luật 50 nước
            else:
                v = get_terminal_value(b)
            for n in reversed(path):
                v = -v * 0.99  # Discount factor: ưu tiên Checkmate sớm (v > 0) và né Checkmate muộn
                n.backprop(v)
            continue
            
        tensor_t = torch.from_numpy(encoder.board_to_tensor(b)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = net(tensor_t)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            v = apply_endgame_heuristic(b, value.item())
            
        node.expand(list(b.legal_moves), probs, b)
        for n in reversed(path):
            v = -v * 0.99  # Suy giảm tương tự cho Value network
            n.backprop(v)
            
    total_N = sum(c.N for c in root.children.values())
    if total_N == 0:
        return list(board.legal_moves)[0].uci(), root_value, []

    # --- Draw Avoider Heuristic ---
    # Lọc ra các nước đi an toàn không dẫn đến hòa ngay lập tức (Stalemate, 3-Fold, 50-move rule)
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
        prob = child.N / total_N
        candidate_moves.append({
            "action": action, 
            "uci": action.uci(), 
            "prob": float(prob)
        })
        
    candidate_moves.sort(key=lambda x: x["prob"], reverse=True)
    
    # Ưu tiên chọn nước có prob cao nhất NHƯNG phải an toàn (không bị hòa)
    best_move = None
    for cand in candidate_moves:
        if cand["action"] in safe_actions:
            best_move = cand["action"]
            break
            
    # Nếu tất cả các nước đều dẫn đến hòa, đành chọn nước mặc định tốt nhất
    if best_move is None:
        best_move = candidate_moves[0]["action"]

    # Để giao diện Web hiểu được AI đã CỐ TÌNH chọn một nước % thấp hơn nhằm tránh hòa,
    # chúng ta sẽ lọc riêng nước đã chọn lên top đầu của list hiển thị.
    top_moves_res = []
    
    # Đưa best_move lên vị trí số 1 (100% hiển thị trực quan cho người dùng thấy engine quyết định lấy nước này)
    best_cand = next((c for c in candidate_moves if c["action"] == best_move), candidate_moves[0])
    top_moves_res.append({"uci": best_cand["uci"], "prob": best_cand["prob"]})
    
    # Đưa các nước còn lại vào list
    for c in candidate_moves:
        if c["action"] != best_move and len(top_moves_res) < 5:
            top_moves_res.append({"uci": c["uci"], "prob": c["prob"]})
            
    return best_move.uci(), root_value, top_moves_res
