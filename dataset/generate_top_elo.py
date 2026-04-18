import os
import glob
import chess.pgn
import h5py
import numpy as np
import time

def process_top_games():
    input_dir = 'dataset/data_2700'
    output_dir = 'dataset/train_top_6m'
    target_positions = 6_000_000

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pgn_files = glob.glob(os.path.join(input_dir, '*.pgn'))
    print(f"Found {len(pgn_files)} PGN files to process in {input_dir}.")

    if len(pgn_files) == 0:
        print("No PGN files found!")
        return

    # Step 1: Scan all headers to extract Elo and offsets
    print("Step 1: Scanning PGN headers to extract Average Elo...")
    start_time = time.time()
    
    game_info = [] # List of tuples: (avg_elo, offset, file_path)

    for pgn_path in pgn_files:
        print(f"Scanning {pgn_path}...")
        with open(pgn_path, 'r', encoding='utf-8') as f:
            while True:
                offset = f.tell()
                try:
                    headers = chess.pgn.read_headers(f)
                except Exception:
                    continue
                    
                if headers is None:
                    break
                    
                if headers.get("Variant", "Standard") != "Standard":
                    continue
                    
                w_elo = headers.get("WhiteElo", "0")
                b_elo = headers.get("BlackElo", "0")
                
                try:
                    w_elo_val = int(w_elo) if w_elo != '?' else 0
                    b_elo_val = int(b_elo) if b_elo != '?' else 0
                    avg_elo = (w_elo_val + b_elo_val) / 2
                    
                    game_info.append((avg_elo, offset, pgn_path))
                except ValueError:
                    pass

    print(f"Total valid games found: {len(game_info)}")
    print(f"Scanning took {time.time() - start_time:.2f} seconds.")

    # Step 2: Sort by Elo descending
    print("Step 2: Sorting games by Elo...")
    game_info.sort(key=lambda x: x[0], reverse=True)

    # Step 3: Extract positions from the top games
    print(f"Step 3: Extracting top games to reach {target_positions} positions...")
    
    fens = []
    moves = []
    results = []
    
    result_map = {
        '1-0': 1,
        '1/2-1/2': 0,
        '0-1': -1,
        '*': 0
    }
    
    total_positions = 0
    games_processed = 0
    
    # We will write in chunks to .h5 files
    chunk_size = 500_000
    chunk_index = 0
    
    def save_chunk():
        nonlocal chunk_index, fens, moves, results
        if len(fens) == 0:
            return
            
        h5_path = os.path.join(output_dir, f'top_elo_chunk_{chunk_index}.h5')
        print(f"Saving chunk {chunk_index} with {len(fens)} positions to {h5_path}...")
        try:
            with h5py.File(h5_path, 'w') as fh:
                fh.create_dataset('fen', data=np.array(fens, dtype='S90'), compression='gzip')
                fh.create_dataset('move', data=np.array(moves, dtype='S5'), compression='gzip')
                fh.create_dataset('value', data=np.array(results, dtype=np.int8), compression='gzip')
        except Exception as e:
            print(f"Failed writing {h5_path}: {e}")
            
        chunk_index += 1
        fens.clear()
        moves.clear()
        results.clear()

    start_time = time.time()
    
    for elo, offset, pgn_path in game_info:
        if total_positions >= target_positions:
            break
            
        with open(pgn_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                continue
                
            if game is None:
                continue
                
            result_str = game.headers.get("Result", "*")
            game_res = result_map.get(result_str, 0)
            
            board = game.board()
            for move in game.mainline_moves():
                if total_positions >= target_positions:
                    break
                    
                relative_res = game_res if board.turn == chess.WHITE else -game_res
                
                fens.append(board.fen().encode('utf-8'))
                moves.append(move.uci().encode('utf-8'))
                results.append(relative_res)
                
                board.push(move)
                total_positions += 1
                
                if len(fens) >= chunk_size:
                    save_chunk()
                    
        games_processed += 1
        if games_processed % 5000 == 0:
            print(f"Processed {games_processed} games... extracted {total_positions} positions")

    # Save any remaining elements
    if len(fens) > 0:
        save_chunk()

    print(f"Extraction complete! Extracted {total_positions} positions tightly from the top {games_processed} highest Elo games.")
    print(f"Lowest Elo included: {game_info[games_processed-1][0] if games_processed > 0 else 'N/A'}")
    print(f"Extraction took {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    process_top_games()
