import numpy as np
import string
import os
import io
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import data_helpers



SEQUENCE_LEN = 200

################## READING functions: BEGIN ################
def pad(x, max_len):    
    for _ in range( max_len - len(x) ):
        x.append('END')
    return x

def read_from_dir(in_fold_dir):
    reviews = []
    for filename in os.listdir(in_fold_dir):
        if filename.endswith(".txt"):
            # read from file containing the text review
            in_filepath = in_fold_dir + filename
            review = []
            with io.open(in_filepath, 'r', encoding='windows-1252') as f:
                review = [str(x) for x in f.read().split()] # get tokens from the file
                # print('review before pad: ', review) # RM
                
                # all reviews must have less than sequence length == SEQUENCE_LEN length
                if len(review) < SEQUENCE_LEN:
                    review = pad(review, max_len=SEQUENCE_LEN)
                else:
                    review = review[:200]
                # print('review after pad: ', review)   
                reviews.append(review)    

    print("LEN: ", len(reviews))              
    return reviews
################## READING functions: END ################


def load_data():
    trustworthy_hotel_positive = read_from_dir('deception_dataset/hotel/positive/truthful/') 
    trustworthy_hotel_negative = read_from_dir('deception_dataset/hotel/negative/truthful/')
    trustworthy_restaurant = read_from_dir('deception_dataset/restaurant/truthful/')
    trustworthy_doctor = read_from_dir('deception_dataset/doctor/truthful/')

    untrustworthy_hotel_positive = read_from_dir('deception_dataset/hotel/positive/deceptive_turker/')
    untrustworthy_hotel_negative = read_from_dir('deception_dataset/hotel/negative/deceptive_turker/') 
    untrustworthy_hotel_expert_positive =  read_from_dir('deception_dataset/hotel/positive/deceptive_expert/')
    untrustworthy_hotel_expert_negative =  read_from_dir('deception_dataset/hotel/negative/deceptive_expert/')
    
    untrustworthy_restaurant = read_from_dir('deception_dataset/restaurant/deceptive_MTurk/')
    untrustworthy_doctor = read_from_dir('deception_dataset/doctor/deceptive_MTurk/')

    TRUSTWORTHY_REVIEWS_DICT = {
        "hotel":trustworthy_hotel_positive + trustworthy_hotel_negative,
        "restaurant":trustworthy_restaurant,
        "doctor":trustworthy_doctor
    }

    UNTRUSTWORTHY_REVIEWS_DICT = { 
        "hotel":untrustworthy_hotel_positive + untrustworthy_hotel_negative,
        "restaurant":untrustworthy_restaurant,
        "doctor":untrustworthy_doctor
    }

    TRUSTWORTHY_REVIEWS  = trustworthy_hotel_positive   + trustworthy_hotel_negative   + trustworthy_restaurant   + trustworthy_doctor
    UNTRUSTWORTHY_REVIEWS = untrustworthy_hotel_positive + untrustworthy_hotel_negative + untrustworthy_restaurant + untrustworthy_doctor
    VOCAB, VOCAB_INV =  data_helpers.build_vocab(TRUSTWORTHY_REVIEWS + UNTRUSTWORTHY_REVIEWS, vocab_size=30001)

    return TRUSTWORTHY_REVIEWS_DICT, UNTRUSTWORTHY_REVIEWS_DICT, VOCAB, VOCAB_INV


TRUSTWORTHY_REVIEWS_DICT, UNTRUSTWORTHY_REVIEWS_DICT, VOCAB, VOCAB_INV = load_data()



#########################################################################################
def get_vocab():
    return VOCAB, VOCAB_INV

def get_reviews_from_dict(reviews_dict, keys = []):
    if not keys:
        raise Exception("Invalid key: key must be one of the following: \'hotel\', \'restaurant\', \'doctor\'")

    reviews = []
    for key in keys:
        if key in reviews_dict:
            reviews += reviews_dict[key]
        else:
            raise Exception("Invalid key: key must be one of the following: \'hotel\', \'restaurant\', \'doctor\'")
    return reviews

def get_reviews(review_type="", keys = []):
    if not review_type:
        raise Exception("Invalid review_type: review_type must be either \'trustworthy\' or \'untrustworthy\'.")
    if not keys:
        raise Exception("Invalid key: key must be one of the follwowing: \'hotel\', \'restaurant\', \'doctor\'")

    if review_type=="trustworthy":
        return get_reviews_from_dict(reviews_dict=TRUSTWORTHY_REVIEWS_DICT, keys=keys)
    elif review_type=="untrustworthy":
        return get_reviews_from_dict(reviews_dict=TRUSTWORTHY_REVIEWS_DICT, keys=keys)
    else:
        raise Exception("Invalid review_type: review_type must be either \'trustworthy\' or \'untrustworthy\'.")




################################################################################################################
def get_data(keys=[], name="cross_domain"):
    if name == "cross_domain":
        Vocab = 
    elif name == "mix_domain":   
        Vocab = 
    else:
        print("Invalid dataset name")

    trustworthy_reviews   = get_reviews(review_type="trustworthy", keys=keys)
    untrustworthy_reviews = get_reviews(review_type="trustworthy", keys=keys)

    reviews = trustworthy_reviews + untrustworthy_reviews
    data = data_helpers.build_input_data(reviews, VOCAB)

    # generaate labels
    labels_trustworthy   = [[1,0] for _ in range(len(trustworthy_reviews   ))]
    labels_untrustworthy = [[0,1] for _ in range(len(untrustworthy_reviews ))]

    labels = np.array(labels_trustworthy + labels_untrustworthy)

    # data_helpers.build_vocab(TRUSTWORTHY_REVIEWS + UNTRUSTWORTHY_REVIEWS, vocab_size=30001)

    print("data len: ", data.shape[0])
    print("labels len: ", labels.shape[0])
        
    return data, labels



######################## CROSS DOMAIN { #######################
def get_cross_domain_dataset(test_keys=[]):
    train_keys = ["hotel"]
    train_data, train_labels = get_data(keys=train_keys, name="cross_domain")
    test_data,  test_labels  = get_data(keys=test_keys, name="cross_domain")
    vocab, vocab_inv = data_helpers.build_vocab(np.vstack((train_data,test_data)), vocab_size=30001)

    return train_data, train_labels, test_data, test_labels, vocab, vocab_inv

def get_cross_domain_dataset_restaurant():
    return get_cross_domain_dataset(test_keys=["restaurant"])

def get_cross_domain_doctor_doctor():
    return get_cross_domain_dataset(test_keys=["doctor"])
######################## CROSS DOMAIN } #######################



######################## MIX DOMAIN { #########################
# from sklearn.model_selection import train_test_split

def get_mix_domain_dataset(fold=1):
    keys = ["hotel", "restaurant", "doctor"]
    data, labels = get_data(keys=keys, name="mix_domain")

    MAX_FOLD = 5
    LEN_DATA = len(data)
    DELTA = int(LEN_DATA / MAX_FOLD)
    start_idx = (fold-1)*DELTA
    end_idx   = start_idx + DELTA

    # np.concatenate
    train_data   = np.concatenate((data[0:start_idx, :],  data[end_idx:,:]))
    train_labels = np.concatenate((labels[0:start_idx:end_idx,:], labels[end_idx:,:]))

    test_data   = data[start_idx:end_idx]
    test_labels = labels[start_idx:end_idx]

    vocab, vocab_inv = data_helpers.build_vocab(data, vocab_size=30001)

    return train_data, train_labels, test_data, test_labels, vocab, vocab_inv
######################## MIX DOMAIN } #########################




# # TESTING {
# train_data, train_labels, test_data, test_labels, vocab, vocab_inv = get_cross_domain_dataset(test_keys=["restaurant"])
# print("cross_domain_dataset_hotel_train len: ", len(train_data))
# print("cross_domain_dataset_hotel_test len: ", len(test_data))

# train_data, train_labels, test_data, test_labels, vocab, vocab_inv = get_cross_domain_dataset(test_keys=["doctor"])
# print("cross_domain_dataset_hotel_train len: ", len(train_data))
# print("cross_domain_dataset_hotel_test len: ", len(test_data))


mixed_domain_dataset  = get_mix_domain_dataset()
# TESTING }