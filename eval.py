import torch
from sklearn import metrics
import pandas as pd
import data
from torch.utils.data import DataLoader
from network import CapsNet_Text
import data_helpers
from w2v import load_word2vec
import argparse
import numpy as np
import json


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='dataset_op_spam_v1.4',
                    help='Options: eurlex_raw_text.p, rcv1_raw_text.p, wiki30k_raw_text.p')
parser.add_argument('--vocab_size', type=int, default=30001, help='vocabulary size')
parser.add_argument('--vec_size', type=int, default=300, help='embedding size')
parser.add_argument('--sequence_length', type=int, default=200, help='the length of documents')
parser.add_argument('--is_AKDE', type=bool, default=True, help='if Adaptive KDE routing is enabled')
parser.add_argument('--num_epochs', type=int, default=120, help='Number of training epochs')
parser.add_argument('--tr_batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for training')
parser.add_argument('--start_from', type=str, default='', help='')

parser.add_argument('--num_compressed_capsule', type=int, default=128, help='The number of compact capsules')
parser.add_argument('--dim_capsule', type=int, default=16, help='The number of dimensions for capsules')

parser.add_argument('--learning_rate_decay_start', type=int, default=0,
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
parser.add_argument('--learning_rate_decay_every', type=int, default=20,
                    help='how many iterations thereafter to drop LR?(in epoch)')
parser.add_argument('--learning_rate_decay_rate', type=float, default=0.95,
                    help='how many iterations thereafter to drop LR?(in epoch)')



args = parser.parse_args()
# params = vars(args)
# print(json.dumps(params, indent = 2))

X_trn, Y_trn, Y_trn_o, X_tst, Y_tst, Y_tst_o, vocabulary, vocabulary_inv = data_helpers.load_data(fold=1)

X_trn = X_trn.astype(np.int32)
X_tst = X_tst.astype(np.int32)
Y_trn = Y_trn.astype(np.int32)
Y_tst = Y_tst.astype(np.int32)

embedding_weights = load_word2vec('glove', vocabulary_inv, num_features=300)
args.num_classes = Y_trn.shape[1]




def eval(fold_ix=1):
    test_data = data.Data(fold=fold_ix, is_test_data=True) # the dataset merges self.real_U_file and self.fake_U_file
    test_data_loader = DataLoader(test_data, batch_size=50, shuffle=True, num_workers=8)
    # self.discriminator(validation_data_loader)
    
    X_trn, Y_trn, Y_trn_o, X_tst, Y_tst, Y_tst_o, vocabulary, vocabulary_inv = data_helpers.load_data(fold=fold_ix)

    X_trn = X_trn.astype(np.int32)
    X_tst = X_tst.astype(np.int32)
    Y_trn = Y_trn.astype(np.int32)
    Y_tst = Y_tst.astype(np.int32)

    args.num_classes = Y_trn.shape[1]

    embedding_weights = load_word2vec('glove', vocabulary_inv, num_features=300)
    
    capsule_net = CapsNet_Text(args, embedding_weights)
    path = 'save/model/fold' + str(fold_ix) + '/nlp-capsule50.pth'
    capsule_net.load_state_dict(torch.load(path))
    # discriminator.eval()

    with torch.no_grad():
        for _, (input, labels) in enumerate(test_data_loader):
            # input, labels = input, labels
            _, preds = capsule_net(input, labels) # (None, vocab_size, sequence_len+1)
            
            # print('preds without round: ' , preds.squeeze(2))

            preds = preds.round().squeeze(2)
            # print('input : ', input)
            # print('preds squeeze 2 : ', preds)
            # print('labels: ', labels)

            # print('preds 0: ' , preds.shape[0])
            # print('preds 1: ' , preds.shape[1])
            # print('labels 0: ', labels.shape[0])
            # print('labels 1: ', labels.shape[1])

        
            report = metrics.classification_report(labels, preds)
            print('report on fold ' + str(fold_ix) + ': ')
            print(report)
            # df = pd.DataFrame(json.dumps(report)).transpose()
            # df.to_csv('report/fold' + str(fold_ix) + '/classification_report.csv', index = False)



for fold in range(1, 6):
    eval(fold_ix=fold)