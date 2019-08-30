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
from torch import autograd
sys.path.append("../../")
from text_utils.text_preprocess import *
#from model import *

GLOVE_PATH = "../../toxic_comment/input/glove.6B.300d.txt"
dat = pd.read_csv("../input/train.csv")

# data is 7920 rows
# columns: id, label, tweet

# seperate x and y
X = dat[['tweet']]
Y = dat['label'].values

# preprocess tweet data
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(GLOVE_PATH)

X['tweet'] = data_process(X, 'tweet')
X_padded = get_indices(X['tweet'].values, word_to_index, 200)

# build glove Matrix
glove_matrix = build_embedding_matrix(word_to_vec_map, word_to_index)



# turn X and Y into torch tensors
x_train_torch = torch.tensor(X_padded, dtype=torch.long)
y_train_torch = torch.tensor(Y, dtype=torch.float32)


# make dataloader
train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)


class BiLstm_Model(nn.Module):
    def __init__(self, embedding_matrix, Dense_Hidden_Units, LSTM_UNITS, Max_Features):
        super(BiLstm_Model, self).__init__()
        embed_size = embedding_matrix.shape[1]
        self.hidden_dim = 128
        self.embedding = nn.Embedding(Max_Features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix,
                                                          dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS*2, LSTM_UNITS)
        self.linear1 = nn.Linear(256, 200)
        self.linear2 = nn.Linear(200, 50)
        self.linear_out = nn.Linear(50, 1)
        self.hidden = self.init_hidden()


    def init_hidden(self):
            return (autograd.Variable(torch.randn(2, 5, self.hidden_dim)),
                                autograd.Variable(torch.randn(2, 5, self.hidden_dim)))
    def forward(self, X):
        self.hidden = self.init_hidden()
        X = self.embedding(X)
        X1, self.hidden = self.lstm1(X, self.hidden)
        X2, self.hidden = self.lstm2(X1)
        avg_pool = torch.mean(X2, 1)
        max_pool, _ = torch.max(X2, 1)
        h_conc = torch.cat((avg_pool, max_pool), 1)
        X4 = F.relu(self.linear1(h_conc))
        X5 = F.relu(self.linear2(X4))
        X6 = F.sigmoid(self.linear_out(X5))
        return X6



# define model Parameters and initialize
LSTM_UNITS = 128
bilstmnet = BiLstm_Model(glove_matrix, 200, LSTM_UNITS, max_features)
error = nn.BCELoss()
optimizer = torch.optim.SGD(bilstmnet.parameters(), lr=.001)

# test model
#x1, xhidden, x2, x4, x5, x6 = bilstmnet(*x_batch)

# train model
loss_list = []
count = 0
iteration_list = []
accuracy_list = []

for batch in train_loader:
    x_batch = batch[:1]
    y_batch = batch[-1]
    optimizer.zero_grad()
    outputs = bilstmnet(*x_batch)
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

