from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        pass

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        pass

class SentimentDataModule(pl.LightningDataModule):
    def __init__(self):
        pass
    
    def setup(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass