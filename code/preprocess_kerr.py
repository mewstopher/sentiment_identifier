from imports import *

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

# define model Parameters and initialize
LSTM_UNITS = 128
bilstmnet = BiLstm_Model(glove_matrix, 200, LSTM_UNITS, max_features)
error = nn.BCELoss()
optimizer = torch.optim.SGD(bilstmnet.parameters(), lr=.001)

def train_mod(model, epochs, print_every=100, save_dir)
    """
    method for training model.

    PARAMS
    -----------------------------
    epochs: number of epochs to trains for
    print_every: how often to print accuracy and loss
    save_dir: directory to save model checkpoints
    """
    # train model
    loss_list = []
    count = 0
    iteration_list = []
    accuracy_list = []
for epochs in range(epochs):
        for batch in train_loader:
            x_batch = batch[:1]
            y_batch = batch[-1]
            optimizer.zero_grad()
            outputs = model(*x_batch)
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
            if count % 300 == 0:
                torch.save(model.state_dict(), save_dir+"{}".format(model))
    return iteration_list, loss_list, accuracy_list


plt.plot(iteration_list, loss_list)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.show()

plt.plot(iteration_list, accuracy_list)
plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.show()


