import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import bci as bci
import os
import pickle


class Config:
    TRAIN_ERROR = 0
    TEST_ERROR = 36
    MODEL_PATH = 'model/mini-project-1.pth.tar'
    LEARNING_RATE = 1e-3
    EPOCHS = 5000


class Net(nn.Module):
    def __init__(self, nb_hidden):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(28, 56, kernel_size=5)
        self.conv2 = nn.Conv1d(56, 112, kernel_size=5)
        self.fc1 = nn.Linear(224, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), kernel_size=4, stride=4))
        x = F.relu(F.max_pool1d(self.conv2(x), kernel_size=3, stride=3))
        x = F.relu(self.fc1(x.view(-1, 224)))
        x = self.fc2(x)
        return x


def save_model(model, path):
    if not path == None:
        torch.save(model, path)


def load_model(path):
    if os.path.exists(path):
        return torch.load(path)
    else:
        return None


def train_model(model, train_input, train_target, model_path=None, learning_rate=1e-3, epochs=5000):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)

    for e in range(epochs):
        output = model(train_input)
        loss = criterion(output, train_target)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)

    save_model(model, model_path)
    

def compute_nb_errors(model, input, target):
    output = model(input)
    _, predicted_classes = output.data.max(1)
    nb_errors = target.ne(predicted_classes).sum()
    return nb_errors


#--- Load data
train_input, train_target = bci.load(root = './data_bci')
train_input, train_target = Variable(train_input).cuda(), Variable(train_target).cuda()

test_input, test_target = bci.load(root = './data_bci', train=False)
test_input, test_target = Variable(test_input).cuda(), Variable(test_target).cuda()


#--- Train mode
model = Net().cuda()
train_model(model, train_input, train_target, 'model/mini-project-1.pth.tar')


#--- Predict mode
model = load_model('model/mini-project-1.pth.tar').cuda()


#--- Statistic 
train_error = compute_nb_errors(model, train_input, train_target)
test_error = compute_nb_errors(model, test_input, test_target)
print('--TRAIN ERROR: {:.2f}% - {} / {}'.format(train_error * 100/ train_target.size(0), train_error, train_target.size(0)))
print('--TEST ERROR: {:.2f}% - {} / {}'.format(test_error * 100 / test_target.size(0), test_error, test_target.size(0)))
