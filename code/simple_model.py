
class SimpleModel(nn.Module):
    def __init__(self, embedding_matrix, Dense_Hidden_Units, LSTM_UNITS, Max_Features):
        super(SimpleModel, self).__init__()
        embed_size = embedding_matrix.shape[1]
        self.embedding = nn.Embedding(Max_Features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix,
                                                          dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, batch_first=True)
        self.linear1 = nn.Linear(128, 200)
        self.linear2 = nn.Linear(200, 50)
        self.linear_out = nn.Linear(50, 1)

    def forward(self, X):
        X = self.embedding(X)
        X, g = self.lstm1(X)
        X = F.relu(self.linear1(g[0]))
        X = F.relu(self.linear2(X))
        X = F.sigmoid(self.linear_out(X))
        return X


