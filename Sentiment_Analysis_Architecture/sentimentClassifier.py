from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer

class SentimentClassifier(pl.LightningModule):
    
    def __init__(self, num_classes, pretrained_model_name="bert-base-uncased"):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        #Visualize the outputs and it's shape
        pooled_outputs = outputs.pooler_output
        #Visualize the pooled_outs and it's shape
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log("train_loss ", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log("validation_loss ", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters, lr=2e-5)
