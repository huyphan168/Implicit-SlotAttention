from torch.utils.data import Dataset, DataLoader

class CoCoDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, index: int) -> dict:
        return {}
    
    def collate_fn(self, batch: list) -> dict:
        return {}
    
    