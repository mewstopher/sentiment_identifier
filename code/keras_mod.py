from imports import *
from text_utils import process

def bidirectional_lstm(input_shape, word_to_vec_map, word_to_index, trainable=False):
    """
    creates the lstm graph using keras

    PARAMS
    ------------------------------------
    input_shape: shape of input (maxlen)
    word_to_vec_map: dictionary mapping words to their embeddings
    word_to_index: dictionary mapping words to their vocab indices
    """

    sentence_indices = Input(shape=input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index, trainable=trainable)
    embeddings = embedding_layer(sentence_indices)

    X = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embeddings)
    X = GlobalMaxPool1D()(X)
    X = Dense(50, activation='relu')(X)
    X = Dropout(.1)(X)
    X = Dense(6, activation='sigmoid')(X)
    model = Model(inputs=sentence_indices, outputs=X)

    return model

def trial_lstm(input_shape, word_to_vec_map, word_to_index, trainable=False):
    """
    creates the lstm graph using keras

    PARAMS
    ------------------------------------
    input_shape: shape of input (maxlen)
    word_to_vec_map: dictionary mapping words to their embeddings
    word_to_index: dictionary mapping words to their vocab indices
    """

    sentence_indices = Input(shape=input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index, trainable=trainable)
    embeddings = embedding_layer(sentence_indices)

    X = Bidirectional(LSTM(200, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embeddings)
    X = LSTM(50,  dropout=0.1)(X)
    X = Dense(50, activation='relu')(X)
    X = Dropout(.1)(X)
    X = Dense(1, activation='sigmoid')(X)
    model = Model(inputs=sentence_indices, outputs=X)

    return model


