import os
# import h5py
import os.path as osp
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data

__all__ = ['EPIDataSetTrain', 'EPIDataSetTest', 'EPIDataSetTrain1', 'EPIDataSetTest1']


class EPIDataSetTrain(data.Dataset):
    def __init__(self, data_tr1, data_tr2, label_tr):
        super(EPIDataSetTrain, self).__init__()
        self.data1 = data_tr1
        self.data2 = data_tr2
        self.label = label_tr

        assert len(self.data1) == len(self.label), \
            "the number of sequences and labels must be consistent."

        print("The number of positive data is {}".format(sum(self.label.reshape(-1) == 1)))
        print("The number of negative data is {}".format(sum(self.label.reshape(-1) == 0)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_one1 = self.data1[index]
        data_one2 = self.data2[index]
        label_one = self.label[index]

        return {"data1": data_one1, "data2": data_one2, "label": label_one}


class EPIDataSetTest(data.Dataset):
    def __init__(self, data_te1, data_te2, label_te):
        super(EPIDataSetTest, self).__init__()
        self.data1 = data_te1
        self.data2 = data_te2
        self.label = label_te

        assert len(self.data1) == len(self.label), \
            "the number of sequences and labels must be consistent."
        print("The number of positive data is {}".format(sum(self.label.reshape(-1) == 1)))
        print("The number of negative data is {}".format(sum(self.label.reshape(-1) == 0)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_one1 = self.data1[index]
        data_one2 = self.data2[index]
        label_one = self.label[index]

        return {"data1": data_one1, "data2": data_one2, "label": label_one}


class EPIDataSetTrain1(data.Dataset):
    def __init__(self, data_tr1, data_tr2, data_tr3, data_tr4, label_tr):
        super(EPIDataSetTrain1, self).__init__()
        self.data1 = data_tr1
        self.data2 = data_tr2
        self.data3 = data_tr3
        self.data4 = data_tr4
        self.label = label_tr

        assert len(self.data1) == len(self.label), \
            "the number of sequences and labels must be consistent."

        print("The number of positive data is {}".format(sum(self.label.reshape(-1) == 1)))
        print("The number of negative data is {}".format(sum(self.label.reshape(-1) == 0)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_one1 = self.data1[index]
        data_one2 = self.data2[index]
        data_one3 = self.data3[index]
        data_one4 = self.data4[index]
        label_one = self.label[index]

        return {"data1": data_one1, "data2": data_one2, "data3": data_one3, "data4": data_one4, "label": label_one}


class EPIDataSetTest1(data.Dataset):
    def __init__(self, data_te1, data_te2, data_te3, data_te4, label_te):
        super(EPIDataSetTest1, self).__init__()
        self.data1 = data_te1
        self.data2 = data_te2
        self.data3 = data_te3
        self.data4 = data_te4
        self.label = label_te

        assert len(self.data1) == len(self.label), \
            "the number of sequences and labels must be consistent."
        print("The number of positive data is {}".format(sum(self.label.reshape(-1) == 1)))
        print("The number of negative data is {}".format(sum(self.label.reshape(-1) == 0)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_one1 = self.data1[index]
        data_one2 = self.data2[index]
        data_one3 = self.data3[index]
        data_one4 = self.data4[index]
        label_one = self.label[index]

        return {"data1": data_one1, "data2": data_one2, "data3": data_one3, "data4": data_one4, "label": label_one}

class EPIDataSetTrain2(data.Dataset):
    def __init__(self, data_tr1, data_tr2, data_tr3, label_tr):
        super(EPIDataSetTrain2, self).__init__()
        self.data1 = data_tr1
        self.data2 = data_tr2
        self.data3 = data_tr3
        self.label = label_tr

        assert len(self.data1) == len(self.label), \
            "the number of sequences and labels must be consistent."

        print("The number of positive data is {}".format(sum(self.label.reshape(-1) == 1)))
        print("The number of negative data is {}".format(sum(self.label.reshape(-1) == 0)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_one1 = self.data1[index]
        data_one2 = self.data2[index]
        data_one3 = self.data3[index]
        label_one = self.label[index]

        return {"data1": data_one1, "data2": data_one2, "data3": data_one3, "label": label_one}


class EPIDataSetTest2(data.Dataset):
    def __init__(self, data_te1, data_te2, data_te3, label_te):
        super(EPIDataSetTest2, self).__init__()
        self.data1 = data_te1
        self.data2 = data_te2
        self.data3 = data_te3
        self.label = label_te

        assert len(self.data1) == len(self.label), \
            "the number of sequences and labels must be consistent."
        print("The number of positive data is {}".format(sum(self.label.reshape(-1) == 1)))
        print("The number of negative data is {}".format(sum(self.label.reshape(-1) == 0)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_one1 = self.data1[index]
        data_one2 = self.data2[index]
        data_one3 = self.data3[index]
        label_one = self.label[index]

        return {"data1": data_one1, "data2": data_one2, "data3": data_one3, "label": label_one}
