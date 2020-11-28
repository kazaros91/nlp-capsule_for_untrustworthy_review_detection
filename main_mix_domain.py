from __future__ import division, print_function, unicode_literals
import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import json
import random
import time
from torch.autograd import Variable
from torch.optim import Adam
from network import CapsNet_Text,BCE_loss
from w2v import load_word2vec
import data_helpers
import datasets


torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='dataset_op_spam_v1.4',
                    help='Options: eurlex_raw_text.p, rcv1_raw_text.p, wiki30k_raw_text.p')
parser.add_argument('--vocab_size', type=int, default=30001, help='vocabulary size')
parser.add_argument('--vec_size', type=int, default=300, help='embedding size')
parser.add_argument('--sequence_length', type=int, default=200, help='the length of documents')
parser.add_argument('--is_AKDE', type=bool, default=True, help='if Adaptive KDE routing is enabled')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
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
params = vars(args)
print(json.dumps(params, indent = 2))



def transformLabels(labels):
    label_index = list(set([l for _ in labels for l in _]))
    label_index.sort()

    variable_num_classes = len(label_index)
    target = []
    for _ in labels:
        tmp = np.zeros([variable_num_classes], dtype=np.float32)
        tmp[[label_index.index(l) for l in _]] = 1
        target.append(tmp)
    target = np.array(target)
    return label_index, target


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


# save Loss and Epochs in csv
def save_loss(Loss, path): 
    num_epochs = len(Loss)
    Epochs=list(range(1, num_epochs+1)) 
    Loss_data = {'Epochs': Epochs, 'Loss': Loss}

    import pandas as pd
    df = pd.DataFrame(Loss_data, columns = ['Epochs', 'Loss'])
    df.to_csv(path)


for FOLD in range(1, 6):
    X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv = datasets.get_mix_domain_dataset(fold=FOLD)

    X_trn = X_trn.astype(np.int32)
    X_tst = X_tst.astype(np.int32)
    Y_trn = Y_trn.astype(np.int32)
    Y_tst = Y_tst.astype(np.int32)

    args.num_classes = Y_trn.shape[1]
    print(len(vocabulary_inv))
    embedding_weights = load_word2vec('glove', vocabulary_inv, num_features=300)
    
    
    capsule_net = CapsNet_Text(args, embedding_weights)
    current_lr = args.learning_rate
    optimizer = Adam(capsule_net.parameters(), lr=current_lr)
    # capsule_net = nn.DataParallel(capsule_net).cuda()


    losses = []
    for epoch in range(args.num_epochs):
        # torch.cuda.empty_cache()

        nr_trn_num = X_trn.shape[0]
        nr_batches = int(np.ceil(nr_trn_num / float(args.tr_batch_size)))

        if epoch > args.learning_rate_decay_start and args.learning_rate_decay_start >= 0:
            frac = (epoch - args.learning_rate_decay_start) // args.learning_rate_decay_every
            decay_factor = args.learning_rate_decay_rate  ** frac
            current_lr = current_lr * decay_factor
        print(current_lr)
        set_lr(optimizer, current_lr)

        capsule_net.train()
        total_loss = 0
        for iteration, batch_idx in enumerate(np.random.permutation(range(nr_batches))):
            start = time.time()
            start_idx = batch_idx * args.tr_batch_size
            end_idx = min((batch_idx + 1) * args.tr_batch_size, nr_trn_num)

            X = X_trn[start_idx:end_idx]
            Y = Y_trn[start_idx:end_idx]
            # data = Variable(torch.from_numpy(X).long()).cuda()
            data = Variable(torch.from_numpy(X).long())
            
            batch_labels, batch_target = transformLabels(Y)
            # batch_target = Variable(torch.from_numpy(batch_target).float()).cuda()
            batch_target = Variable(torch.from_numpy(batch_target).float())
            Y = Variable(torch.from_numpy(Y).float())

            optimizer.zero_grad()
            print('data: ', data)
            print('labels: ', Y)
            print('len data: ', len(data))
            print('len labels: ', len(Y))

            poses, activations = capsule_net(data, Y)
            loss = BCE_loss(activations, Y)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            done = time.time()
            elapsed = done - start
            # print("Labels: {}, batch_labels: {}, batch_target: {},  predictions: {} ".format(Y, batch_labels, batch_target, activations.squeeze(2)))
            print("labels: {},  predictions: {} ".format(Y, activations.squeeze(2)))
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f} {:.5f}".format(
                        (iteration+1), nr_batches,
                        (iteration+1) * 100 / nr_batches,
                        loss.item(), elapsed),
                        end="")
            total_loss = total_loss + loss.item()
        losses.append(total_loss/float(nr_batches))

        # torch.cuda.empty_cache()
        

        # if not created mkdir
        if (epoch + 1) > 20:
            checkpoint_path = os.path.join('save/model/mix_domain/fold' + str(FOLD), 'nlp-capsule'  + str(epoch + 1) + '.pth')
            torch.save(capsule_net.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))

            checkpoint_path = os.path.join('save/loss/mix_domain/fold'  + str(FOLD), 'nlp-capsule_loss' + str(epoch + 1) + '.csv')
            save_loss(losses, checkpoint_path)
            print("model saved to {}".format(checkpoint_path))