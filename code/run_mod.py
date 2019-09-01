from imports import *
from train_mod import *

GLOVE_PATH = "../../toxic_comment/input/glove.6B.200d.txt"
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

# define model Parameters and initialize
max_features = len(word_to_index) + 1
LSTM_UNITS = 128
bilstmnet = BiLstm_Model(glove_matrix, 200, LSTM_UNITS, max_features)
error = nn.BCELoss()
optimizer = torch.optim.SGD(bilstmnet.parameters(), lr=.001)

iteration_list, loss_list, accuracy_list = train_mod(train_loader, bilstmnet,
                                                     optimizer, error, "../output/bilstmnet")

