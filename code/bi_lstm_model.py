from imports import *

class BiLstm_Model(nn.Module):
    def __init__(self, embedding_matrix, Dense_Hidden_Units, LSTM_UNITS, Max_Features):
        super(BiLstm_Model, self).__init__()
        embed_size = embedding_matrix.shape[1]
        self.hidden_dim = 128
        self.embedding = nn.Embedding(Max_Features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix,
                                                          dtype=torch.float32))
        self.embedding.weight.requires_grad = True
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


