import sys
sys.path.append('..')

import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import utils as ut
import pickle

class kneeMRIDatasetk(data.Dataset):
    def __init__(self, root_dir, train=1, k=1, transform=False, weights=None):
        super().__init__()
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

        if self.train == 1:
            self.records = pd.read_csv(self.root_dir + 'train_k{0}.csv'.format(k), header=None, names=['aclDiagnosis', 'examId',
                                                                                       'seriesNo', 'kneeLR',
                                                                                       'roiX', 'roiY', 'roiZ', 'roiHeight',
                                                                                       'roiWidth','roiDepth', 'volumeFilename'])
        elif self.train == 0:
            self.records = pd.read_csv(self.root_dir + 'val_k{0}.csv'.format(k), header=None, names=['aclDiagnosis', 'examId',
                                                                                       'seriesNo', 'kneeLR',
                                                                                       'roiX', 'roiY', 'roiZ', 'roiHeight',
                                                                                       'roiWidth','roiDepth', 'volumeFilename'])

        else:
            self.records = pd.read_csv(self.root_dir + 'test_k.csv', header=None, names=['aclDiagnosis', 'examId',
                                                                                       'seriesNo', 'kneeLR',
                                                                                       'roiX', 'roiY', 'roiZ', 'roiHeight',
                                                                                       'roiWidth','roiDepth', 'volumeFilename'])

        self.paths = [self.root_dir + filename for filename in self.records['volumeFilename'].tolist()]
        self.labels = self.records['aclDiagnosis'].tolist()
        for i, l2 in enumerate(self.labels):
            if l2 == 2:
                self.labels[i] = 1

        if self.train == 1:
            if weights is None:
                neg_count = self.labels.count(0)
                pos_count = self.labels.count(1)
                neg_weight = (1 / (neg_count / ((neg_count + pos_count) / 2)))
                pos_weight = (1 / (pos_count / ((neg_count + pos_count) / 2)))
                self.weights = torch.FloatTensor([neg_weight, pos_weight])
            else:
                self.weights = torch.FloatTensor(weights)
        else:
            self.weights = torch.FloatTensor([1, 1])


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with open(self.paths[index], 'rb') as file_handler:
            input = pickle.load(file_handler)

        label = torch.FloatTensor([self.labels[index]])

        weight = torch.FloatTensor([self.weights[self.labels[index]]])

        if self.transform:
            input = ut.random_shift(input, 25)
            input = ut.random_rotate(input, 25)
            input = ut.random_flip(input)

        input = ut.center_resize(input, 256)
        input = (input - 328.7806) / 251.4325
        input = np.stack((input,)*3, axis=1)
        input = torch.FloatTensor(input)

        return input, label, weight

class MRNetDataset(data.Dataset):

    def __init__(self, root_dir, task, plane, train=True, transform=False, weights=None):
        super().__init__()
        self.task = task
        self.plane = plane
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

        if self.train:
            self.folder_path = self.root_dir + 'train/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'train-{0}.csv'.format(task), header=None, names=['id', 'label'])
        else:
            self.folder_path = self.root_dir + 'valid/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'valid-{0}.csv'.format(task), header=None, names=['id', 'label'])

        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
        self.paths = [self.folder_path + filename +
                      '.npy' for filename in self.records['id'].tolist()]
        self.labels = self.records['label'].tolist()

        if self.train:
            if weights is None:
                neg_count = self.labels.count(0)
                pos_count = self.labels.count(1)
                neg_weight = (1 / (neg_count / ((neg_count + pos_count) / 2)))
                pos_weight = (1 / (pos_count / ((neg_count + pos_count) / 2)))
                self.weights = torch.FloatTensor([neg_weight, pos_weight])
            else:
                self.weights = torch.FloatTensor(weights)
        else:
            self.weights = torch.FloatTensor([1, 1])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        input = np.load(self.paths[index])

        label = torch.FloatTensor([self.labels[index]])

        weight = torch.FloatTensor([self.weights[self.labels[index]]])

        if self.transform:
            input = ut.random_shift(input, 25)
            input = ut.random_rotate(input, 25)
            input = ut.random_flip(input)

        input = (input - 58.2274) / 48.1512
        input = np.stack((input,)*3, axis=1)
        input = torch.FloatTensor(input)

        return input, label, weight

