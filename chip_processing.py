# coding:utf-8
import os.path as osp  # os.path模块主要用于获取文件的属性
import os
import argparse
import numpy as np
from Bio import SeqIO

# SEQ_LEN = 201  # 选取的序列段长度
SEQ_LEN = 101  # 选取的序列段长度
OFFSET = 3000
INDEX = ['chr' + str(i + 1) for i in range(25)]
INDEX[22] = 'chrM'  # 列表INDEX存放染色体的编号 [chr1,chr2,chr3,...,chrM,chrX,chrY] 总共有22条常染色体，加上线粒体/X染色体/Y染色体
INDEX[23] = 'chrX'
INDEX[24] = 'chrY'
CHROM = {}  # 读取每条染色体的总碱基数目  key---chr1 value---249250621 键值对
convscore1 = {}  # 存放染色体的conv得分的起始位置 key---chr1 value---11392 键值对
convscore2 = {}  # 存放染色体的conv得分列表  key----chr value---[0.0064,0.0051,...,0.78] 键值对
gsmscore1 = {}
gsmscore2 = {}
ebhscore1 = {}
ebhscore2 = {}


with open('/home/sc3/users/xu/MotifC/hg19/chromsize') as f:
    for i in f:
        line_split = i.strip().split()  # strip()函数用于删除空白符 而split()用于分割，默认空白符
        if line_split[0] not in INDEX:
            continue
        CHROM[line_split[0]] = int(line_split[1])  # key:chr1   value:249250621  "染色体号——染色体长度"键值对

path = "/home/sc3/users/xu/MotifC/convscore"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
for file in files:
    position = path + '/' + file  # 构造绝对路径
    #  print(position)
    with open(position, "r", encoding='utf-8') as f:
        data = f.readline().strip().split(' ')  # 读取第一行数据得到染色体上conv得分的起始位置
        convscore1[data[1][6:]] = int(data[2][6:])

        seq = []  # 接着读取染色体上的进化信息
        # if data[1][6:] == "chrM":
        while True:
            line = f.readline()
            # 如果没有读到数据，跳出循环
            if not line:
                break
            if len(line) > 10:
                continue
            seq.append(float(line[:-1]))
        mininlist = min(seq)
        maxinlist = max(seq)
        for i in seq:
            i = (i - mininlist) / (maxinlist - mininlist)
        convscore2[data[1][6:]] = seq

"""
for key in convscore1.keys():
    print(key, convscore1[key])
for key in convscore2.keys():
    print(key, convscore2[key], sep="\n")
"""


#  将每一个碱基序列进行编码one-hot编码
def one_hot(sequence_dict, chrom, start, end):
    seq_dict = {'A': [1, 0, 0, 0], 'G': [0, 0, 1, 0],
                'C': [0, 1, 0, 0], 'T': [0, 0, 0, 1],
                'a': [1, 0, 0, 0], 'g': [0, 0, 1, 0],
                'c': [0, 1, 0, 0], 't': [0, 0, 0, 1]}
    temp1 = []
    temp2 = []
    seq = str(sequence_dict[chrom].seq[start:end])  # 碱基序列的起始和结束位点
    startconv = convscore1[chrom]  # 染色体上conv得分的起始位置
    li = convscore2[chrom][start - startconv:end - startconv]
    index = 0
    for c in seq:
        temp1.append(seq_dict.get(c, [0, 0, 0, 0]))
        if index >= len(li):
            temp2.append(0)
        else:
            temp2.append(li[index])
        index += 1
    return temp1, temp2


def pos_location(chr, start, end, resize_len):
    """
    chr:染色体编号
    start:染色体上某一序列的起始位置
    end:染色体上某一序列的终止位置
    resize_len:在[start,end]序列段上选择的子序列长度
    """
    original_len = end - start
    if original_len < resize_len:
        start_update = start - np.ceil((resize_len - original_len) / 2)
    elif original_len > resize_len:
        start_update = start + np.ceil((original_len - resize_len) / 2)
    else:
        start_update = start

    end_update = start_update + resize_len
    if end_update > CHROM[chr]:
        end_update = CHROM[chr]
        start_update = end_update - resize_len
    return int(start_update), int(end_update)


def neg_location(peak, resize_len, offset):
    peak_r = peak - offset
    if peak_r < 0:
        print("peak_r get minus index")
        peak_r = 0
    start = peak_r
    end = peak_r + resize_len
    return start, end


def get_data(seqs_bed, sequence_dict):
    """
    seqs_bed:bed文件，标示序列段的起始和结束位置 文件的前三列数据是 chr1	878407	878870
    sequence_dict:字典类型 key:chr1 value:AGCT碱基序列 键值对
    """
    seqs1 = []  # one-hot编码
    seqs2 = []  # 进化信息
    labels = []
    lines = open(seqs_bed).readlines()  # 碱基序列的起始和结束位置 readlines()函数读取所有文件内容
    index = list(range(len(lines)))
    # np.random.shuffle(index)
    for i in index:
        line_split = lines[i].strip().split()  # 第一个数据是染色体编号(chr1) 第二个数据是碱基序列的起始位置 第三个是结束位置
        chr = line_split[0]  # 染色体编号 chr1
        if chr not in INDEX:
            continue
        start, end = int(line_split[1]), int(line_split[2])  # 碱基序列起始和结束位点 878407	878870
        start_p, end_p = pos_location(chr, start, end, SEQ_LEN)  # 在这段序列上选取长度为SEQ_LEN的子序列
        seqs1.append(one_hot(sequence_dict, chr, start_p, end_p)[0])  # 将序列进行one-hot编码
        seqs2.append(one_hot(sequence_dict, chr, start_p, end_p)[1])
        labels.append(1)  # 正例
        peak = int((start_p + end_p) / 2)  # 中间位置peak
        start_n, end_n = neg_location(peak, SEQ_LEN, OFFSET)  # SEQ_LEN=201 OFFSET=3000
        seqs1.append(one_hot(sequence_dict, chr, start_n, end_n)[0])
        seqs2.append(one_hot(sequence_dict, chr, start_n, end_n)[1])
        labels.append(0)  # 反例

    seqs1 = np.array(seqs1, dtype=np.float32)
    seqs2 = np.array(seqs2, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    seqs1 = seqs1.transpose((0, 2, 1))  # 对数据维度进行调整 变成4*N的矩阵
    b, _, _ = seqs1.shape
    seqs2 = np.reshape(seqs2, (b, 1, SEQ_LEN))
    return seqs1, seqs2, labels


def get_args():
    parser = argparse.ArgumentParser(description="pre-process data.")
    # parser.add_argument("-d", dest="dir", type=str, default='')
    # parser.add_argument("-n", dest="name", type=str, default='')
    parser.add_argument("-d", dest="dir", type=str, default='A549_01/ATF3')
    parser.add_argument("-n", dest="name", type=str, default='ATF3')

    return parser.parse_args()


def main():
    params = get_args()
    print(params)
    name = params.name
    data_dir = params.dir
    out_dir = osp.join(params.dir, 'data/')  # 将目录和文件名合成一个路径
    if not osp.exists(out_dir):
        os.mkdir(out_dir)  # os.mkdir()方法用于以数字权限模式创建目录
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open('hg19/hg19.fa'), 'fasta'))
    print('Experiment on %s dataset' % name)  # 在哪个数据集上，比如ATF3数据集
    seqs_bed = data_dir + '/%s_peak1.bed' % name  # 文件路径A594/ATF3/ATF3_peak1.bed
    print(seqs_bed)
    with open(seqs_bed, "r") as df:
        print(df.readline())
    # pfmfile = data_dir + '/%s.txt' % name  # 文件路径A594/ATF3/ATF3.txt
    seqs1, seqs2, labels = get_data(seqs_bed, sequence_dict)  # , pfmfile

    np.savez(out_dir + '%s_data.npz' % name, data1=seqs1, data2=seqs2, label=labels)


if __name__ == '__main__':
    main()
