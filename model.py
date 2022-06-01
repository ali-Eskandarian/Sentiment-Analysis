import torch
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
import torch.nn as nn

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup


class bertlstm(pl.LightningModule):
  def __init__(self,bert_name):
    super().__init__()
    self.bert = BertModel.from_pretrained(bert_name, return_dict=True)
    self.lstm = nn.LSTM(768, 32, batch_first=True,bidirectional=True)
    self.linear1 = nn.Linear(32*2, 256)
    self.linear2 = nn.Linear(32*2, 3)

  def forward(self, ids, mask):
    x = self.bert(
               ids,
               attention_mask=mask)
    x, _ = self.lstm(x.pooler_output)
    # x = self.linear1(x)
    x = self.linear2(x)
    x = torch.sigmoid(x)
    output = torch.sigmoid(x)
    return output

  def training_step(self, batch, batch_idx):
      input_ids = batch["input_ids"]
      attention_mask = batch["attention_mask"]
      y_hat = self(input_ids,attention_mask)
      loss = nn.CrossEntropyLoss(y_hat, batch["targets"])
      return loss

  def configure_optimizers(self):
      return torch.optim.AdamW(self.parameters(), lr=2e-5)