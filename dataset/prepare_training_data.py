import os
import glob
import chess.pgn
import h5py
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def process_pgn_to_h5(pgn_path: str, output_dir: str):
    """
    Reads a single PGN file, extracts FEN, UCI move, and Game Result 
    for each position, and saves them into an HDF5 file.
    """
    base_name = os.path.basename(pgn_path).replace('.pgn', '.h5')
    h5_path = os.path.join(output_dir, base_name)
    
    # Skip if we already processed this file
    if os.path.exists(h5_path):
        return
    
    fens = []
    moves = []
    results = []
    
    # Map for parsed result. +1 = White wins, 0 = Draw, -1 = Black wins
    result_map = {
        '1-0': 1,
        '1/2-1/2': 0,
        '0-1': -1,
        '*': 0 # Unknown or aborted, treated as draw.
    }
    
    # We'll buffer in memory and write at the end. 
    # Usually a single monthly 2700+ elite file might have 10,000-50,000 games.
    # We'll batch write if needed, but keeping in RAM (a few million elements) per process is typically okay.
    with open(pgn_path, 'r', encoding='utf-8') as pgn:
        while True:
            # We want to catch exceptions for corrupted games
            try:
                game = chess.pgn.read_game(pgn)
            except Exception:
                # E.g. ValueError for impossible moves
                game = None
                
            if game is None:
                break
                
            # If variant is not standard, skip it
            if game.headers.get("Variant", "Standard") != "Standard":
                continue
                
            result_str = game.headers.get("Result", "*")
            game_res = result_map.get(result_str, 0)
            
            board = game.board()
            for move in game.mainline_moves():
                # Value perspective: relative to the side to move
                # 1 if current side eventually wins, -1 if current side loses
                # Often used in AlphaZero / Leela Chess training
                relative_res = game_res if board.turn == chess.WHITE else -game_res
                
                fens.append(board.fen().encode('utf-8'))
                moves.append(move.uci().encode('utf-8'))
                results.append(relative_res)
                
                board.push(move)

    if not fens:
        return
        
    # Write to chunked/compressed HDF5 arrays for efficiency
    try:
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('fen', data=np.array(fens, dtype='S90'), compression='gzip')
            f.create_dataset('move', data=np.array(moves, dtype='S5'), compression='gzip')
            f.create_dataset('value', data=np.array(results, dtype=np.int8), compression='gzip')
    except Exception as e:
        print(f"\nFailed writing {h5_path}: {e}")
        # Clean up partial corrupted files
        if os.path.exists(h5_path):
            os.remove(h5_path)

def _worker(args):
    """Wrapper function for multiprocessing starmap/imap"""
    process_pgn_to_h5(*args)

def main():
    input_dir = 'data_2700'
    output_dir = 'train_data'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pgn_files = glob.glob(os.path.join(input_dir, '*.pgn'))
    print(f"Found {len(pgn_files)} PGN files to process in {input_dir}.")
    
    if len(pgn_files) == 0:
        print("No PGN files found. Please make sure the path is correct.")
        return

    # Use CPU cores minus 1 to save some responsiveness
    num_cores = max(1, mp.cpu_count() - 1)
    print(f"Starting parsing jobs on {num_cores} parallel cores...", flush=True)
    
    args = [(path, output_dir) for path in pgn_files]
    
    with mp.Pool(num_cores) as pool:
        for _ in tqdm(pool.imap_unordered(_worker, args), total=len(args), desc="Processing PGNs to HDF5"):
            pass

    print("Data extraction complete! The HDF5 files are saved in:", os.path.abspath(output_dir))

if __name__ == '__main__':
    main()
