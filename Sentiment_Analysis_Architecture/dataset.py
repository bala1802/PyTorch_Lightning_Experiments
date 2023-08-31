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
        input_ids = encoded["input_ids"].squeeze() #Visualize the shape of input_ids before and after squeezing
        attention_mask = encoded["attention_mask"].squeeze() #Visualize the shape of attention_mask before and after squeezing
        return input_ids, attention_mask, label

class SentimentDataModule(pl.LightningDataModule):
    def __init__(self, train_texts, train_labels, val_texts, val_labels, tokenizer, batch_size, max_length):
        super(SentimentDataModule, self).__init__()
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.val_texts = val_texts
        self.val_labels = val_labels
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
    
    def setup(self):
        self.train_dataset = CustomDataset(self.train_texts, self.train_labels, self.tokenizer, self.max_length)
        self.val_dataset = CustomDataset(self.val_texts, self.val_labels, self.tokenizer, self.max_length)    

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)