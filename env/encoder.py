import chess
import numpy as np

# --- Move Encoding ---

FILES = "abcdefgh"
RANKS = "12345678"
PROMOTIONS = ['q', 'r', 'n', 'b']

# Generate all possible UCI strings
ALL_UCI_MOVES = []
# 1. Normal moves
for from_file in FILES:
    for from_rank in RANKS:
        for to_file in FILES:
            for to_rank in RANKS:
                move = f"{from_file}{from_rank}{to_file}{to_rank}"
                ALL_UCI_MOVES.append(move)

# 2. Pawn promotions (Rank 7->8 for white, Rank 2->1 for black)
# White promotions
for from_file in FILES:
    for to_file in FILES:
        if abs(FILES.index(from_file) - FILES.index(to_file)) <= 1:
            for p in PROMOTIONS:
                ALL_UCI_MOVES.append(f"{from_file}7{to_file}8{p}")
                
# Black promotions
for from_file in FILES:
    for to_file in FILES:
        if abs(FILES.index(from_file) - FILES.index(to_file)) <= 1:
            for p in PROMOTIONS:
                ALL_UCI_MOVES.append(f"{from_file}2{to_file}1{p}")

# Add underpromotions?
# The list now contains all valid UCI strings. Length should be 4096 + 3*24*4 = ? Wait. 4096 + (8 straight + 14 cross)*4 * 2 = 4096 + 176 = 4272.
# Let's clean it up slightly and make dictionaries.

ALL_UCI_MOVES = list(set(ALL_UCI_MOVES)) # Just in case
ALL_UCI_MOVES.sort()

MOVE_TO_INDEX = {move: i for i, move in enumerate(ALL_UCI_MOVES)}
INDEX_TO_MOVE = {i: move for move, i in MOVE_TO_INDEX.items()}
NUM_MOVES = len(ALL_UCI_MOVES)

class Encoder:
    """
    Encodes chess board state and actions for Neural Network input.
    """
    def __init__(self):
        self.num_channels = 18

    def board_to_tensor(self, board: chess.Board) -> np.ndarray:
        """
        Converts a chess.Board into a (18, 8, 8) float32 numpy array.
        Channels:
        0-5: White pieces (P, N, B, R, Q, K)
        6-11: Black pieces
        12: Turn (1 if white, 0 if black)
        13: White Kingside castling
        14: White Queenside castling
        15: Black Kingside castling
        16: Black Queenside castling
        17: En-passant square (1 at the square if available, 0 otherwise)
        """
        tensor = np.zeros((18, 8, 8), dtype=np.float32)
        
        # Pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                color = int(piece.color) # 1 for White, 0 for Black
                piece_type = piece.piece_type - 1 # 0 to 5
                
                # Channel calculation: White is 0-5, Black is 6-11
                channel = piece_type if color == 1 else piece_type + 6
                rank, file = chess.square_rank(square), chess.square_file(square)
                tensor[channel, rank, file] = 1.0
                
        # Turn
        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0
            
        # Castling Rights
        if board.has_kingside_castling_rights(chess.WHITE): tensor[13, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): tensor[14, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): tensor[15, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): tensor[16, :, :] = 1.0
        
        # En-passant
        if board.ep_square is not None:
            rank, file = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
            tensor[17, rank, file] = 1.0
            
        return tensor

    def fen_to_tensor(self, fen: str) -> np.ndarray:
        board = chess.Board(fen)
        return self.board_to_tensor(board)

    def move_to_index(self, move: chess.Move) -> int:
        uci = move.uci()
        return MOVE_TO_INDEX.get(uci, 0) # Fallback to 0 if invalid

    def move_str_to_index(self, uci: str) -> int:
        return MOVE_TO_INDEX.get(uci, 0)

    def index_to_move(self, index: int) -> chess.Move:
        uci = INDEX_TO_MOVE[index]
        return chess.Move.from_uci(uci)

if __name__ == "__main__":
    enc = Encoder()
    board = chess.Board()
    t = enc.board_to_tensor(board)
    print(f"Tensor shape: {t.shape}")
    print(f"Number of possible moves: {NUM_MOVES}")
    print(f"Index for e2e4: {enc.move_str_to_index('e2e4')}")
