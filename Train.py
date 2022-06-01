#libraries

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import shutil
import sys
from Datasets import dataset_pt
from model import bertlstm


def create_data_loader(self, df, tokenizer, max_len, batch_size):
  ds = dataset_pt(reviews=df.tweet.to_numpy(), targets=df.label.to_numpy(), tokenizer=tokenizer, max_len=max_len)
  return DataLoader(ds, batch_size=batch_size, num_workers=2)


class Train_data():
    def __init__(self,Datatrain_,Dataval_,Datatest_, tokenizer, max_token_len, BATCH_SIZE,model):
        self.model = model
        self.train_data_loader=create_data_loader(Datatrain_, tokenizer, max_token_len, BATCH_SIZE)
        self.val_data_loader = create_data_loader(Dataval_, tokenizer, max_token_len, BATCH_SIZE)
        self.test_data_loader = create_data_loader(Datatest_, tokenizer, max_token_len, BATCH_SIZE)

    def train_epoch(self, loss_fn, optimizer, device, n_examples):
      self.model = self.model.train()
      losses = []
      correct_predictions = 0

      for d in self.train_data_loader:
          input_ids = d["input_ids"].to(device)
          attention_mask = d["attention_mask"].to(device)
          targets = d["targets"].to(device)

          outputs = self.model(
            ids=input_ids,
            mask=attention_mask
          )

          _, preds = torch.max(outputs, dim=1)
          loss = loss_fn(outputs, targets)

          correct_predictions += torch.sum(preds == targets)
          losses.append(loss.item())

          loss.backward()
          nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
          optimizer.step()
          optimizer.zero_grad()
      return correct_predictions.double() / n_examples, np.mean(losses)

    def eval_model(self, loss_fn, device, n_examples):
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0
        with torch.no_grad():
          for d in self.val_data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = self.model(
              ids=input_ids,
              mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)
    def test_model(self, loss_fn, device, n_examples):
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0
        with torch.no_grad():
          for d in self.test_data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = self.model(
              ids=input_ids,
              mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)

    def get_predictions(self, device, data_loader):
        self.model = self.model.eval()

        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []

        with torch.no_grad():
          for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = self.model(
              ids=input_ids,
              mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        return review_texts, predictions, prediction_probs, real_values

