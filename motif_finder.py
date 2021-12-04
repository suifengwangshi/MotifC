#!/usr/bin/python

import os
import sys
import time
import argparse
import math
import numpy as np
import os.path as osp

import torch
from torch.utils.data import DataLoader

# custom functions defined by user
from datasets import EPIDataSetTrain, EPIDataSetTest
from utils import Dict
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    """FPN for semantic segmentation"""
    def __init__(self, motiflen=15):
        super(FCN, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # classifier head
        c_in = 832
        self.linear1 = nn.Linear(c_in, 64)
        self.drop = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(64, 1)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        score = out1
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        skip4 = out1
        # classifier
        out2 = skip4.view(b, -1)
        out2 = self.linear1(out2)
        out2 = self.relu(out2)
        out2 = self.drop(out2)
        out2 = self.linear2(out2)
        out_class = self.sigmoid(out2)

        return out_class, score[0]


def motif(device, model, state_dict, train_loader, test_loader, outdir):
    # loading model parameters
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    motif_data = [0.] * kernel_num
    for i_batch, sample_batch in enumerate(train_loader):
        X_data = sample_batch["data"].float().to(device)
        with torch.no_grad():
            _, score_p = model(X_data)
        data = X_data[0].data.cpu().numpy()
        score_p = score_p.data.cpu().numpy()
        max_index = np.argmax(score_p, axis=1)
        for i in range(kernel_num):
            index = max_index[i]
            data_slice = data[:, index:(index+motifLen)]
            motif_data[i] += data_slice

    pfm = compute_pfm(motif_data)
    writeFile(pfm, 'train_all', outdir)
    # for test data
    motif_data = [0.] * kernel_num
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(device)
        with torch.no_grad():
            _, score_p = model(X_data)
        data = X_data[0].data.cpu().numpy()
        score_p = score_p.data.cpu().numpy()
        max_index = np.argmax(score_p, axis=1)
        for i in range(kernel_num):
            index = max_index[i]
            data_slice = data[:, index:(index + motifLen)]
            motif_data[i] += data_slice

    pfm = compute_pfm(motif_data)
    writeFile(pfm, 'test_all', outdir)


def compute_pfm(motifs):
    pfms = []
    for motif in motifs:
        sum_ = np.sum(motif, axis=0)
        pfm = motif / sum_
        pfms.append(pfm)

    return pfms


def writeFile(pfm, flag, outdir):
    out_f = open(outdir + '/{}_pfm.txt'.format(flag), 'w')
    out_f.write("MEME version 5.1.1\n\n")
    out_f.write("ALPHABET= ACGT\n\n")
    out_f.write("strands: + -\n\n")
    out_f.write("Background letter frequencies\n")
    out_f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
    for i in range(len(pfm)):
        out_f.write("MOTIF " + "{}\n".format(i+1))
        out_f.write("letter-probability matrix: alength= 4 w= {} nsites= {}\n".format(motifLen, motifLen))
        current_pfm = pfm[i]
        for col in range(current_pfm.shape[1]):
            for row in range(current_pfm.shape[0]):
                out_f.write("{:.4f} ".format(current_pfm[row, col]))
            out_f.write("\n")
        out_f.write("\n")
    out_f.close()


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="FCN for motif location")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the training data.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of a specified data.")
    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")
    parser.add_argument("-o", dest="outdir", type=str, default='./motifs/',
                        help="Where to save experimental results.")

    return parser.parse_args()


args = get_args()
motifLen_dict = Dict(os.getcwd() + '/motifLen.txt')
motifLen = motifLen_dict[args.name]
kernel_num = 64


def main():
    """Create the model and start the training."""
    if torch.cuda.is_available():
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
    Data = np.load(osp.join(args.data_dir, '%s_data.npz' % args.name))
    seqs, label = Data['data'], Data['label']
    ##
    pos_index = (label == 1)
    seqs = seqs[pos_index]
    label = label[pos_index]
    ##
    cv_num = 5
    interval = int(len(seqs) / cv_num)
    index = range(len(seqs))
    # 5-fold cross validation
    for cv in range(1):
        index_test = index[cv*interval:(cv+1)*interval]
        index_train = list(set(index) - set(index_test))
        # build training data generator
        data_tr = seqs[index_train]
        label_tr = label[index_train]
        train_data = EPIDataSetTrain(data_tr, label_tr)
        train_loader = DataLoader(train_data, batch_size=1, num_workers=1)
        # build test data generator
        data_te = seqs[index_test]
        label_te = label[index_test]
        test_data = EPIDataSetTest(data_te, label_te)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
        # Load weights
        checkpoint_file = osp.join(args.checkpoint, 'model_best%d.pth' % cv)
        chk = torch.load(checkpoint_file)
        state_dict = chk['model_state_dict']
        model = FCN(motiflen=motifLen)
        motif(device, model, state_dict, train_loader, test_loader, args.outdir)


if __name__ == "__main__":
    main()

