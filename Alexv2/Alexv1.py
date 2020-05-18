import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from emotionData import TrainDataset, SampleTrainDataset, TestDataset, SampleTestDataset
import matplotlib.pyplot as plt
import time

batch_size = 500
epochs = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    torch.cuda.empty_cache()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, 8, stride=1)  #
        self.pool1 = nn.MaxPool2d(2, stride=1)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=6, stride=1)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=5, stride=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1)
        self.fc = nn.Linear(384 * 11 * 11, 7)
        # self.fc = nn.Linear(96*5*5, 7)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 384 * 11 * 11)
        x = F.relu(self.fc(x))
        return x


training_set = TrainDataset(device)
training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

test_set = TestDataset(device)
test_generator = data.DataLoader(test_set, batch_size=batch_size)
print("Data loaded")
print("Number of training samples",training_set.__len__())

net = Net()
if torch.cuda.is_available():
    net.cuda()
learning_rate = 0.01
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_function = nn.CrossEntropyLoss()

train_losses = []
train_accuracies = []
test_accuracies = []

print('Start training')
for i in range(epochs):
    start_time = time.time()
    net.train()
    total_loss = 0
    correct = 0

    for local_batch, local_labels in training_generator:
        output = net(local_batch.float())
        _, predicted_labels = torch.max(output, 1)
        loss = loss_function(output, local_labels.long())
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (predicted_labels == local_labels).sum().item()

    train_loss = total_loss / training_set.__len__()
    train_losses.append(train_loss)
    print("Loss: ", train_loss)
    train_accuracy = correct / training_set.__len__()
    train_accuracies.append(train_accuracy)
    print("Accuracy: ", train_accuracy)

    net.eval()
    with torch.no_grad():
        correct = 0
        for local_batch, local_labels in test_generator:
            output = net(local_batch.float())
            _, predicted_labels = torch.max(output, 1)
            correct += (predicted_labels == local_labels).sum().item()
        test_acc = correct / test_set.__len__()
        test_accuracies.append(test_acc)
        print("Test accuracy: ", test_acc)
    print("Epoch " + str(i) + " completed in: " + str(time.time() - start_time) + " seconds")
    torch.save(net.state_dict(), "checkpoints/model"+str(i)+".pt")


plt.plot(train_losses)
plt.title("Loss")
plt.xlabel("Epochs")
plt.savefig("Loss.png")

plt.clf()
plt.plot(train_accuracies, label='Train')
plt.plot(test_accuracies, label="Test")
plt.xlabel("Epochs")
plt.title("Accuracy")
plt.legend()
plt.savefig("Accuracy.png")
