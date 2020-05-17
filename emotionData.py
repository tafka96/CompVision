import torch
from torch.utils import data
import numpy as np

trainX = 'train_X.npy'
trainY = 'train_Y.npy'

testX = 'test_X.npy'
testY = 'test_Y.npy'


class TrainDataset(data.Dataset):
    def __init__(self, device):
        train_X = np.load(trainX)
        train_Y = np.load(trainY)

        train_X = train_X.reshape(28709, 1, 48, 48)
        train_X = train_X / 255
        x = torch.from_numpy(train_X).to(device)
        y = torch.from_numpy(train_Y).to(device)
        self.labels = y
        self.images = x

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = self.images[index]
        y = self.labels[index]
        return X, y


class TestDataset(data.Dataset):
    def __init__(self, device):
        train_X = np.load(testX)
        train_Y = np.load(testY)

        train_X = train_X.reshape(7178, 1, 48, 48)
        train_X = train_X / 255
        x = torch.from_numpy(train_X).to(device)
        y = torch.from_numpy(train_Y).to(device)
        self.labels = y
        self.images = x

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = self.images[index]
        y = self.labels[index]
        return X, y


class SampleTrainDataset(data.Dataset):
    def __init__(self, device):
        train_X = np.load(trainX)
        train_Y = np.load(trainY)

        train_X = train_X[:200]
        train_Y = train_Y[:200]

        train_X = train_X.reshape(200, 1, 48, 48)
        train_X = train_X / 255
        x = torch.from_numpy(train_X).to(device)
        y = torch.from_numpy(train_Y).to(device)
        self.labels = y
        self.images = x

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = self.images[index]
        y = self.labels[index]
        return X, y

class SampleTestDataset(data.Dataset):
    def __init__(self, device):
        test_X = np.load(testX)
        test_Y = np.load(testY)

        test_X = test_X[:150]
        test_Y = test_Y[:150]


        test_X = test_X.reshape(150, 1, 48, 48)
        test_X = test_X / 255
        x = torch.from_numpy(test_X).to(device)
        y = torch.from_numpy(test_Y).to(device)
        self.labels = y
        self.images = x

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = self.images[index]
        y = self.labels[index]
        return X, y