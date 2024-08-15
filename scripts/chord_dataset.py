import torch

from torch.utils.data import Dataset

class ChordDataset(Dataset):
    """Dataset class for chord fragments."""
    def __init__(self, chord_fragments):
        self.chord_fragments = chord_fragments

    def __len__(self):
        return len(self.chord_fragments)

    def __getitem__(self, idx):
        chord = torch.tensor(self.chord_fragments[idx], dtype=torch.long)
        return chord