import numpy as np
import string
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


SEQUENCE_LEN = 200


def pad(x, max_len):    
    for _ in range( max_len - len(x) ):
        x.append('END')
    return x

def read_from_dir(in_fold_dir):
    reviews = []
    for filename in os.listdir(in_fold_dir):
        reviews.append(review)
        if filename.endswith(".txt"):
            # read from file containing the text review
            in_filepath = in_fold_dir + filename
            review = []
            with open(in_filepath, 'r') as f:
                review = [str(x) for x in f.read().split()] # get tokens from the file
                # print('review before pad: ', review) # RM
                
                # all reviews must have less than sequence length == SEQUENCE_LEN length
                if len(review) < SEQUENCE_LEN:
                    review = pad(review, max_len=SEQUENCE_LEN)
                    reviews.append(review)    
                    # print('review after pad: ', review)   
                else:
                    continue

    print("LEN: ", len(reviews))              
    return reviews


def load_data():
    # DATASET_NAME = 'deception_dataset/'
    data = []
    trustworthy_hotel_positive = read_from_dir('deception_dataset/hotel/positive/truthful/') 
    trustworthy_hotel_negative = read_from_dir('deception_dataset/hotel/negative/truthful/')
    trustworthy_restaurant     = read_from_dir('deception_dataset/restaurant/truthful/')
    trustworthy_doctor         = read_from_dir('deception_dataset/doctor/truthful/')

    untrustworthy_hotel_positive = read_from_dir('deception_dataset/hotel/positive/deceptive/')
    untrustworthy_hotel_negative = read_from_dir('deception_dataset/hotel/negative/deceptive/')
    untrustworthy_restaurant     = read_from_dir('deception_dataset/restaurant/deceptive/')
    untrustworthy_doctor         = read_from_dir('deception_dataset/doctor/deceptive/')

    trustworthy   = trustworthy_hotel_positive   + trustworthy_hotel_negative   + trustworthy_restaurant   + trustworthy_doctor
    untrustworthy = untrustworthy_hotel_positive + untrustworthy_hotel_negative + untrustworthy_restaurant + untrustworthy_doctor
    # trustworthy   = clean(trustworthy)
    # untrustworthy = clean(untrustworty)

    return trustworthy, untrustworthy


import data_helpers
class MixedDomainDataset(Dataset):
    def __init__(self, train_keys=[], test_keys=[]): # change real_U_file to fake_U_file
        super(MixedDomainDataset, self).__init__()

        trustworthy_reviews, untrustworthy_reviews = load_data()
        reviews = trustworthy_reviews + untrustworthy_reviews

        # generate labels
        labels_trustworthy   = [[1,0] for _ in range(len(trustworthy_reviews   ))]
        labels_untrustworthy = [[0,1] for _ in range(len(untrustworthy_reviews ))]

        self.labels = np.array(labels_trustworthy + labels_untrustworthy)

        # convert word2idx
        vocabulary, vocabulary_inv = data_helpers.build_vocab(trustworthy_reviews + untrustworthy_reviews, vocab_size=30001)
        self.data = torch.as_tensor(data_helpers.build_input_data(reviews, vocabulary))

        print("data len: ",   self.data.shape[0])
        print("labels len: ", self.labels.shape[0])
        
        # assert(self.data.shape[0], self.labels.shape[0])

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return self.data.shape[0] 