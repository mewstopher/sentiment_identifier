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

# attempting transfer learning here
model = load_model("../../toxic_comment/output/bidirection_2epoch")

model.summary()
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.summary()

x = Dense(50, activation='relu')(model.layers[-1].output)
o = Dense(1, activation='sigmoid')(x)
model2 = Model(input=model.input, output=[o])
model2.summary()

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.fit(X_padded, Y, epochs=5, batch_size=32, shuffle=True, validation_split=.1)

model.save("../output/model2_keras")

