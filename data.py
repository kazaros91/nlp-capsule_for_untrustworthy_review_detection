import numpy as np
import string
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from files import real_T_file_, real_U_file_


FOLD = 1 # set FOLD number
MAX_FOLD = 5
SEQUENCE_LEN = 200

################## READING functions: BEGIN ################
def read_from_dir(file, fold_ix=FOLD, max_fold = MAX_FOLD, is_test_data=False):
    data = []
    if (is_test_data):
        in_fold_dir = file + 'fold' + str(fold_ix) + '/'
        data = read_from_fold(in_fold_dir, data)
    else:
        for fold in range(1, max_fold+1):
            in_fold_dir = file + 'fold' + str(fold_ix) + '/'
            if fold != fold_ix:
                data = read_from_fold(in_fold_dir, data)
    return data

def pad(x, max_len):    
    for _ in range( max_len - len(x) ):
        x.append('END')
    return x

def read_from_fold(in_fold_dir, data):
    for filename in os.listdir(in_fold_dir):
        if filename.endswith(".txt"):
            # read from file containing the text review
            in_filepath = in_fold_dir + filename
            review = []
            with open(in_filepath, 'r') as f:
                review = [str(x) for x in f.read().split()] # get tokens from the file
                # print('review before pad: ', review) # RM
                
                # all reviews must have less than sequence length == SEQUENCE_LEN length
                if len(review) < SEQUENCE_LEN:
                    data.append(review)    
                    review = pad(review, max_len=SEQUENCE_LEN)
                    # print('revieww after pad: ', review)   
                  
    return data
################## READING functions: END ################


################## LOAD DATA: BEGIN ################
def load_data(fold=FOLD, is_test_data=False):
    # genuine_reviews_for_training is an array of genuine reviews for training. Each review is padded to the "SEQ_LENGTH" with "ENDING_WORD".
    genuine_reviews_for_training   =  read_from_dir(file=real_T_file_, fold_ix=fold, max_fold=5, is_test_data=is_test_data)
    genuine_reviews_for_testing    =  read_from_dir(file=real_T_file_, fold_ix=fold, max_fold=5, is_test_data=(not is_test_data) )
    deceptive_reviews_for_training =  read_from_dir(file=real_U_file_, fold_ix=fold, max_fold=5, is_test_data=is_test_data)
    deceptive_reviews_for_testing   =  read_from_dir(file=real_U_file_, fold_ix=fold, max_fold=5, is_test_data=(not is_test_data))
    return genuine_reviews_for_training, genuine_reviews_for_testing, deceptive_reviews_for_training, deceptive_reviews_for_testing
################## LOAD DATA: END ################


import data_helpers
class Data(Dataset):
    def __init__(self, positive_file=real_T_file_, negative_file=real_U_file_, fold=FOLD, is_test_data=False): # change real_U_file to fake_U_file
        super(Data, self).__init__()
        self.fold = fold


        ###### BEGIN #####
        # load data
        trustworthy_reviews_for_training, trustworthy_reviews_for_testing, untrustworthy_reviews_for_training, untrustworthy_reviews_for_testing = load_data(fold=fold)
        train = trustworthy_reviews_for_training + untrustworthy_reviews_for_training
        test  = trustworthy_reviews_for_testing + untrustworthy_reviews_for_testing

        # generaate labels
        train_labels1 = [[1,0] for _ in range(len(trustworthy_reviews_for_training   ))]
        train_labels0 = [[0,1] for _ in range(len(untrustworthy_reviews_for_training ))]
        test_labels1  = [[1,0] for _ in range(len(trustworthy_reviews_for_testing    ))]
        test_labels0  = [[0,1] for _ in range(len(untrustworthy_reviews_for_testing  ))]

        train_labels = np.array(train_labels1 + train_labels0)
        test_labels  = np.array(test_labels1  + test_labels0)

        # convert word2idx
        vocabulary, vocabulary_inv = data_helpers.build_vocab(train + test, vocab_size=30001)
        train = torch.as_tensor(data_helpers.build_input_data(train, vocabulary))
        test  = torch.as_tensor(data_helpers.build_input_data(test,  vocabulary))
    
        assert(train.shape[0], train_labels.shape[0])
        assert(test.shape[0],  test_labels.shape[0])

        if not is_test_data:
            self.data  = train
            self.labels = train_labels
        else:
            self.data  = test
            self.labels = test_labels
        ###### END ######

        
        ###### ALTERNATIVE: BEGIN ######
        # trustworthy_data   = torch.as_tensor(read_from_dir(positive_file, fold_ix = self.fold, is_test_data = is_test_data))
        # untrustworthy_data = torch.as_tensor(read_from_dir(negative_file, fold_ix = self.fold, is_test_data = is_test_data))

        
        # # [1, 0], [0, 1] labels for trustworthy_data, untrustworthy_data, respecively
        # N_trustworthy, N_untrustworthy = trustworthy_data.shape[0], untrustworthy_data.shape[0]
        
        # trustworthy_labels   = torch.as_tensor([ [1, 0] for _ in range(N_trustworthy)  ])
        # untrustworthy_labels = torch.as_tensor([ [0, 1] for _ in range(N_untrustworthy)]) 
        
        # self.data   = torch.cat((trustworthy_data,   untrustworthy_data),   dim=0)
        # self.labels = torch.cat((trustworthy_labels, untrustworthy_labels), dim=0)
        ###### ALTERNATIVE: END ######


    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return self.data.shape[0] 
################## READING functions: END ################