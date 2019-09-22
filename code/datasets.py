from imports import *
from torch.utils.data import Dataset
import pandas as pd
import torch.functional as F

class SentimentDataset(Dataset):
    """
    class for creating dataset
    """
    def __init__(self, datafile_path, glove_path):
        self.df = pd.read_csv(datafile_path)
        print("successfully run data from {}".format(datafile_path))
        print("number of tweets: {}".format(self.df.shape[1]))
        self.vocab_path = os.path.join(os.path.dirname(datafile_path), "vocab.npy")
        self.vocab_vectors_path = os.path.join(os.path.dirname(datafile_path), "vocab_vectors.npy")
        self.word_to_index, self.index_to_word, self.word_to_vec_map = self.read_glove_vecs(glove_path)
        self.emb_dim = np.int(self.word_to_vec_map['fox'][0])
        self._build_vocab()
        self.max_tweet_len = len(max(self.df.tweet))

    def __len__(self):
        return len(self.df)

    def _build_vocab(self):
        # if previously built vocab exists, use it
        if os.path.isfile(self.vocab_path) and os.path.isfile(self.vocab_vectors_path):
            print('using pre-built vocabulary')
            self.vocab = np.load(self.vocab_path, allow_pickle=True).item()
            self.initial_embeddings = np.load(self.vocab_vectors_path, allow_pickle=True).item
            self.unk_index = self.vocab['unk']
        print("no prebuilt vocab found. building vocab... ")
        embeddings = []
        self.vocab = {}
        embeddings.append(np.zeros(self.emb_dim,))
        token_count = 0
        token_count +=1
        unk_encountered = False
        list_of_tweets = self.df['tweet'].apply(lambda x: text_to_word_sequence(x))
        words_in_sample = [tweet for tweets in list_of_tweets for tweet in tweets]
        for word in words_in_sample:
            if word not in self.vocab:
                if word in self.word_to_index.keys():
                    self.vocab[word] = token_count
                    token_count += 1
                    embeddings.append(self._vec(word))
                else:
                    if not unk_encountered:
                        embeddings.append(self._vec('unk'))
                        self.unk_index = token_count
                        self.vocab['unk'] = self.unk_index
                        token_count +=1
                        unk_encountered = True
                    self.vocab[word] = self.unk_index

        if not unk_encountered:
            embeddings.append(self._vec('unk'))
            self.unk_index = token_count
            token_count +=1
            unk_encountered = True
            self.vocab[word] = self.unk_index
        self.initial_embeddings = np.array(embeddings)
        print('finished building vocab')

        np.save(self.vocab_path, self.vocab)
        np.save(self.vocab_vectors_path, self.initial_embeddings)




    def _vec(self, w):
        return self.word_to_vec_map[w]
    def _tokenize_content(self, data):
        """
        build embedding matrix from glove embeddings
        I want 2 main things here: glove embedings matrix,
        the padded and indexed tweets
        """
        X_tokenized = data['tweet'].apply(lambda x: text_to_word_sequence(x))
        return X_tokenized

    def read_glove_vecs(self, glove_path):
        """
        returns word to indices, index to words, and word
        to glove mappings
        """
        with open(glove_path, 'r') as f:
            words = set()
            word_to_vec_map = {}
            for line in f:
                line = line.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            i = 1
            word_to_index = {}
            index_to_word = {}
            for w in sorted(words):
                word_to_index[w] = i
                index_to_word[i] = w
                i = i + 1
        return word_to_index, index_to_word, word_to_vec_map


    def __getitem__(self, idx):
        data_sample = self.df.iloc[idx]
        target = data_sample['label']
        tweet_indices = self._tokenize_content(data_sample)
        tweet_indices = [self.vocab.get(i, self.unk_index) for i in tweet_indices]
        tweet_len = len(tweet_indices)
        tweet_indices_padded = F.pad(tweet_indices, (0, self.max_tweet_len - tweet_len), mode='constant', value=0)

        return tweet_indices_padded, targets


