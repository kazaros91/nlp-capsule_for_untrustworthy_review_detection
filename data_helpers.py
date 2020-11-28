import numpy as np
import os
import re
import itertools
import scipy.sparse as sp
from collections import Counter
import data
import torch
# from nltk.corpus import stopwords

# cachedStopWords = stopwords.words("english")

SEQUENCE_LEN = 200


''' RM
def clean_str(string):
    # remove stopwords
    # string = ' '.join([word for word in string.split() if word not in cachedStopWords])
    return string
'''

'''
def pad_sentences(sentences, padding_word="<PAD/>", max_length=200):
    sequence_length = min(max(len(x) for x in sentences), max_length)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < max_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:max_length]
        padded_sentences.append(new_sentence)
    return padded_sentences
'''

def build_vocab(sentences, vocab_size=50000):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    # append <UNK/> symbol to the vocabulary
    vocabulary['<UNK/>'] = len(vocabulary)
    vocabulary_inv.append('<UNK/>')
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    x = np.array([[vocabulary[word] if word in vocabulary else vocabulary['<UNK/>'] for word in sentence] for sentence in sentences])
    return x


def load_data(fold = 1, vocab_size=30001):
    # Load data
    trustworthy_reviews_for_training, trustworthy_reviews_for_testing, untrustworthy_reviews_for_training, untrustworthy_reviews_for_testing = data.load_data(fold=fold)
    train = trustworthy_reviews_for_training + untrustworthy_reviews_for_training
    test  = trustworthy_reviews_for_testing + untrustworthy_reviews_for_testing

    # generate labels
    train_labels1 = [[1,0] for _ in range(len(trustworthy_reviews_for_training   ))]
    train_labels0 = [[0,1] for _ in range(len(untrustworthy_reviews_for_training ))]
    test_labels1  = [[1,0] for _ in range(len(trustworthy_reviews_for_testing    ))]
    test_labels0  = [[0,1] for _ in range(len(untrustworthy_reviews_for_testing  ))]

    train_labels = np.concatenate((np.array(train_labels1), np.array(train_labels0)), axis=0)
    test_labels  = np.concatenate((np.array(test_labels1),  np.array(test_labels0)),  axis=0)

    # convert word2idx
    vocabulary, vocabulary_inv = build_vocab(train + test, vocab_size=vocab_size)
    train = build_input_data(train, vocabulary)
    test = build_input_data(test,   vocabulary)
    
    # assert(train.shape[0], train_labels.shape[0])
    # assert(test.shape[0],  test_labels.shape[0] )
    
    return train, train_labels, train_labels, train, test_labels, test_labels, vocabulary, vocabulary_inv


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]