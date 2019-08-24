import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import text, sequence
import sys
import os
import time
import gc
import random
#from tqdm._tqdm_notebook import tqdm_notebook as tqdm
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
sys.path.append("../../")
from text_utils.process import *
from model import *

GLOVE_PATH = "../../toxic_comment/input/glove.6B.100d.txt"
dat = pd.read_csv("../input/train.csv")

# data is 7920 rows
# columns: id, label, tweet

# seperate x and y
X = dat[['tweet']]
Y = dat['label'].values

# preprocess tweet data

# remove punctuation -preprocess is a simple function removing special characters
X = preprocess(X['tweet'])

# tokenize tweet text using text module from keras
# pad text sequences using sequence module from keras
# define max number of features (words)
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(X))
X = tokenizer.texts_to_sequences(X)
X = sequence.pad_sequences(X, maxlen= 220)

max_features = None
max_features = max_features or len(tokenizer.word_index) +1

# Build word embedding matrix using the tokenized tweet text and glove 
# embeddings
# we are going to use a custom function (build_matrix) that also uses a custom 
# function (load_embeddings)
# results are a matrix of embeddings(23852, 100), and 13542 unkown words
glove_matrix, unkown_words = build_matrix(tokenizer.word_index, GLOVE_PATH)

# turn X and Y into torch tensors
x_train_torch = torch.tensor(X, dtype=torch.long)
y_train_torch = torch.tensor(Y, dtype=torch.float32)


# make dataloader
train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)

# define model Parameters and initialize
LSTM_UNITS = 128
simplenet = SimpleModel(glove_matrix, 200, LSTM_UNITS, max_features)
error = nn.BCELoss()
optimizer = torch.optim.SGD(simplenet.parameters(), lr=.0001)

# train model
loss_list = []
count = 0
iteration_list = []
accuracy_list = []

for batch in train_loader:
    x_batch = batch[:1]
    y_batch = batch[-1]
    optimizer.zero_grad()
    outputs = simplenet(*x_batch)
    loss = error(outputs, y_batch)
    loss.backward()
    optimizer.step()
    count += 1
    loss_list.append(loss)
    iteration_list.append(count)
    accuracy = int(((outputs>.5).float().squeeze() == y_batch).sum())/5
    accuracy_list.append(accuracy)
    if count % 100 == 0:
        print('iteration: {} loss: {} accuracy: {}'.format(count, loss, sum(accuracy_list)/(count)))

