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
from collections import defaultdict
from Datasets import dataset_pt
from model import bertlstm
from Train import Train_data
from sklearn.metrics import confusion_matrix
RANDOM_SEED = 17
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
pl.seed_everything(RANDOM_SEED)

data= pd.read_csv("processed_data.csv") # pre-processed data
data=data.drop(columns = ["Unnamed: 0"])

#bert model
Bert = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(Bert)

train, val = train_test_split(data, test_size=0.2)
val, test = train_test_split(val, test_size=0.5)

#find maxlen of tweets
token_lens = []
for txt in data.tweet:
  tokens = tokenizer.encode(txt, max_length=512)
  token_lens.append(len(tokens))
sns.distplot(token_lens)
plt.xlim([0, 256])
plt.xlabel('Token count')
plt.show()
max_token_len = 100

#
BATCH_SIZE = 12
EPOCHS = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = bertlstm(Bert)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
loss_fn = nn.CrossEntropyLoss().to(device)
training = Train_data(train, val, test, tokenizer, max_token_len, BATCH_SIZE)

history = defaultdict(list)
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 5)

    train_acc, train_loss = training.train_epoch(loss_fn, optimizer=optimizer, device=device, n_examples=len(train))

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = training.eval_model(loss_fn, device=device, n_examples=len(train))

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
    history['train_acc'].append(float(train_acc))
    history['train_loss'].append(float(train_loss))
    history['val_acc'].append(float(val_acc))
    history['val_loss'].append(float(val_loss))

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

test_acc, _ = training.test_model(loss_fn, device=device, n_examples=len(test))
print(test_acc.item())

y_review_texts, y_pred, y_pred_probs, y_test = training.get_predictions(device,test)
class_names=['Hate', 'Offensive', 'Normal']
print(classification_report(y_test, y_pred, target_names=class_names))
cf_matrix= confusion_matrix(y_test, y_pred, labels=None)
sns.heatmap(cf_matrix, annot=True)
plt.show()