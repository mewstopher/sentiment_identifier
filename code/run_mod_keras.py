from imports import *
from keras_mod import *

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


# run keras model
model = trial_lstm((200,), word_to_vec_map, word_to_index, trainable=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_padded, Y, epochs=1, batch_size=5, shuffle=True)

