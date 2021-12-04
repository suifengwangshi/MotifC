import numpy as np
import pyBigWig
import os.path as osp

# Data = np.load('./A549_01/ATF3/data/ATF3_data.npz')
# data = Data['data']
# label = Data['label']
#
# print(' data[0] = ', data[0])
# print(' data.shape = ', data.shape)
#
# print(' label[0] = ', label[0])
# print(' label.shape = ', label.shape)

# path_dnase = './bigwig/GM12878.bigwig'
# bw_dnase = pyBigWig.open(path_dnase)
# d_dnase = bw_dnase.chroms()
# # print(d_dnase)
# for chrom in d_dnase:
#     if chrom == 'chr1':
#         idx = 0
#         for entry in bw_dnase.intervals(chrom):
#             start, end, pos = entry
#             print(entry)
#             if idx > 10:
#                 break
#             idx += 1
#     else:
#         continue


f = open(osp.join('./models_rnn', 'record.txt'), 'w')
f.write('CV\ta1\ta2\tAUC\tPRAUC\n')
