import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from env.encoder import Encoder

class ChessDataset(Dataset):
    """
    Standard PyTorch Dataset for reading preprocessed HDF5 files.
    """
    def __init__(self, data_dir: str, max_files: int = None):
        """
        data_dir: Path to the directory containing .h5 files
        """
        self.data_dir = data_dir
        self.h5_files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
        if max_files is not None:
            self.h5_files = self.h5_files[:max_files]
            
        self.encoder = Encoder()
        
        # Pre-compute the sizes of all files so we can map an index to a specific file and row
        self.file_sizes = []
        self.cumulative_sizes = []
        self.total_size = 0
        
        print(f"Indexing {len(self.h5_files)} HDF5 files. This may take a moment...")
        for f in self.h5_files:
            try:
                with h5py.File(f, 'r') as h5f:
                    if 'fen' in h5f:
                        size = len(h5f['fen'])
                        self.file_sizes.append(size)
                        self.total_size += size
                        self.cumulative_sizes.append(self.total_size)
                    else:
                        print(f"Warning: {f} has no 'fen' dataset. Skipping.")
                        self.file_sizes.append(0)
                        self.cumulative_sizes.append(self.total_size)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                self.file_sizes.append(0)
                self.cumulative_sizes.append(self.total_size)
                
        print(f"Total dataset size: {self.total_size} positions.")
        
        # We will keep a small cache of open HDF5 file handles if using single worker,
        # but with multiprocessing Dataloader, we should open on the fly or cautiously.
        # It's safer to open/close or use worker_init_fn.
        self.current_h5 = None
        self.current_f_idx = -1

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Find which file this index belongs to
        # np.searchsorted is fast for this
        f_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        
        # Calculate the local index within that file
        if f_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[f_idx - 1]
            
        # Open file if not open or if we switched files
        if self.current_f_idx != f_idx or self.current_h5 is None:
            if self.current_h5 is not None:
                self.current_h5.close()
            self.current_h5 = h5py.File(self.h5_files[f_idx], 'r')
            self.current_f_idx = f_idx
            
        fen_bytes = self.current_h5['fen'][local_idx]
        move_bytes = self.current_h5['move'][local_idx]
        value = self.current_h5['value'][local_idx]
        
        fen = fen_bytes.decode('utf-8')
        move_str = move_bytes.decode('utf-8')
        
        tensor = self.encoder.fen_to_tensor(fen)
        move_idx = self.encoder.move_str_to_index(move_str)
        
        # value is from -1 to 1, return as float32
        return torch.from_numpy(tensor), torch.tensor(move_idx, dtype=torch.long), torch.tensor(value, dtype=torch.float32)

    def __del__(self):
        if self.current_h5 is not None:
            self.current_h5.close()

def get_dataloader(data_dir: str, batch_size: int, num_workers: int = 4, max_files: int = None):
    dataset = ChessDataset(data_dir, max_files)
    # Important: num_workers > 0 causes issues with h5py handles if not handled properly.
    # We open on the fly, but h5py doesn't like fork. worker_init_fn could help.
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, # Shuffle mixes positions across the whole current file and neighboring, not perfectly global but okay
        num_workers=num_workers,
        pin_memory=True
    )
    return loader

if __name__ == "__main__":
    # Test loader
    loader = get_dataloader("dataset/train_data", batch_size=4, num_workers=0, max_files=1)
    for batch_idx, (tensors, moves, values) in enumerate(loader):
        print("Tensors shape:", tensors.shape)
        print("Moves shape:", moves.shape, moves)
        print("Values shape:", values.shape, values)
        break

