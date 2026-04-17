import h5py
import sys

def verify_h5(file_path):
    print(f"Reading {file_path}...")
    try:
        with h5py.File(file_path, 'r') as f:
            fens = f['fen']
            moves = f['move']
            values = f['value']
            
            print(f"Length of FENs: {len(fens)}")
            print(f"Length of Moves: {len(moves)}")
            print(f"Length of Values: {len(values)}")
            
            if len(fens) > 0:
                print("\nSample Data (First 3 entries):")
                for i in range(min(3, len(fens))):
                    fen = fens[i].decode('utf-8')
                    move = moves[i].decode('utf-8')
                    val = values[i]
                    print(f"[{i}] FEN: {fen}")
                    print(f"    MOVE: {move}")
                    print(f"    VALUE: {val}")

                    
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_h5.py <path_to_h5>")
    else:
        verify_h5(sys.argv[1])
