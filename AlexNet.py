import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
train_X = np.load('train_X.npy')
train_Y = np.load('train_Y.npy')
test_X = np.load('test_X.npy')
test_Y = np.load('test_Y.npy')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, 8, stride=4)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.fc = nn.Linear(96*5*5, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 96 * 5 * 5)
        x = F.relu(self.fc(x))
        return x


net = Net()
learning_rate = 0.01
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()
train_X = train_X.reshape(28709, 1, 48, 48)
train_X = train_X/255
X = torch.from_numpy(train_X)
Y = torch.from_numpy(train_Y)

for i in range(100):
    step = 256
    start = 0
    stop = start + step
    tot_loss = 0
    while start<train_X.shape[0]:
        batch_x = X[start:stop]

        optimizer.zero_grad()
        y_pred = net(batch_x.float())
        losses = loss(y_pred, Y[start:stop].long())
        tot_loss += losses
        losses.backward()
        optimizer.step()

        start = start + step
        stop = stop + step
        if stop > train_X.shape[0]:
            stop = train_X.shape[0]
    print(tot_loss)