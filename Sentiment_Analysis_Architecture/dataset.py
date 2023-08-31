from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max-length",
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        return input_ids, attention_mask, label

class SentimentDataModule(pl.LightningDataModule):
    def __init__(self):
        pass
    
    def setup(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass