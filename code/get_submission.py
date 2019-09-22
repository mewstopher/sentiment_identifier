from imports import *
from keras_mod import *

GLOVE_PATH = "../../toxic_comment/input/glove.6B.300d.txt"
sample_sub = pd.read_csv("../input/sample_submission.csv")
# load in trained model from aws
model = load_model("../output/model2_keras_aws")

model.summary()

#model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model2.fit(X_padded, Y, epochs=5, batch_size=32, shuffle=True, validation_split=.1)

# read in test csv
test = pd.read_csv("../input/test.csv")

# seperate x and y
X = test[['tweet']]
#Y = test['label'].values

# preprocess tweet data

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(GLOVE_PATH)

X['tweet'] = data_process(X, 'tweet')
X_padded = get_indices(X['tweet'].values, word_to_index, 200)

y_test = model.predict([X_padded], batch_size=1024, verbose=1)
sample_sub['label'] = y_test
sample_sub['label'] = sample_sub['label'].map(lambda x: 1 if x>.5 else 0)
sample_sub.to_csv("../output/sample_sub1.csv", index=False)

