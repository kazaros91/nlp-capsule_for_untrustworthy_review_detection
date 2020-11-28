import numpy as np
import string
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


SEQUENCE_LEN = 200

################## READING functions: BEGIN ################
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

    print("LEN: ", len(reviews))              
    return reviews
################## READING functions: END ################


def load_data():
    # DATASET_NAME = 'deception_dataset/'
    trustworthy_hotel_positive = read_from_dir('deception_dataset/hotel/positive/truthful/') 
    trustworthy_hotel_negative = read_from_dir('deception_dataset/hotel/negative/truthful/')
    trustworthy_restaurant = read_from_dir('deception_dataset/restaurant/truthful/')
    trustworthy_doctor = read_from_dir('deception_dataset/doctor/truthful/')

    untrustworthy_hotel_positive = read_from_dir('deception_dataset/hotel/positive/deceptive/')
    untrustworthy_hotel_negative = read_from_dir('deception_dataset/hotel/negative/deceptive/')
    untrustworthy_restaurant = read_from_dir('deception_dataset/restaurant/deceptive/')
    untrustworthy_doctor = read_from_dir('deception_dataset/doctor/deceptive/')

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


def get_vocab():
    return VOCAB

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




###############################################
import data_helpers
class ReviewDataset(Dataset):
    def __init__(self, keys=[]): # change real_U_file to fake_U_file
        super(ReviewDataset, self).__init__()

        trustworthy_reviews   = get_reviews(review_type="trustworthy", keys=keys)
        untrustworthy_reviews = get_reviews(review_type="trustworthy", keys=keys)

        reviews = trustworthy_reviews + untrustworthy_reviews
        self.data = torch.as_tensor(data_helpers.build_input_data(reviews, vocabulary))

        # generaate labels
        labels_trustworthy   = [[1,0] for _ in range(len(trustworthy_reviews   ))]
        labels_untrustworthy = [[0,1] for _ in range(len(untrustworthy_reviews ))]

        self.labels = np.array(labels_trustworthy + labels_untrustworthy)

        # convert word2idx
        vocabulary = get_vocab()

        print("data len: ", self.data.shape[0])
        print("labels len: ", self.labels.shape[0])
        
        # assert(self.data.shape[0], self.labels.shape[0])
        ###### END ######

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return self.data.shape[0] 





### CROSS DOMAIN ###
def get_cross_domain_dataset(test_keys=[]):
    train_keys = ["hotel", "restaurant", "doctor"]
    for test_key in test_keys:
        train_keys.remove(test_key)
    return ReviewDataset(keys=train_keys), ReviewDataset(keys=test_keys)

cross_domain_dataset_hotel_train = ReviewDataset(keys=["restaurant", "doctor"])
cross_domain_dataset_hotel_test  = ReviewDataset(keys=["hotel"])

cross_domain_dataset_restaurant_train = ReviewDataset(keys=["hotel", "doctor"])
cross_domain_dataset_restaurant_test  = ReviewDataset(keys=["restaurant"])

cross_domain_dataset_doctor_train = ReviewDataset(keys=["hotel", "restaurant"])
cross_domain_dataset_doctor_test  = ReviewDataset(keys=["restaurant"])
### CROSS DOMAIN ###


### MIX DOMAIN ###
mixed_domain_dataset  = ReviewDataset(keys=["hotel", "restaurant", "doctor"])
def get_mixed_domain_dataset():
    return mixed_domain_dataset 
### MIX DOMAIN ###









