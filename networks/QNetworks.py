import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self,x):
        return x.view(x.size()[0],-1)

class DQN_fc(nn.Module):
    def __init__(self,stateShape,numActions):
        super(DQN_fc, self).__init__()
        self.fc1 = nn.Linear(stateShape[0],256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,numActions)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN_image(nn.Module):
    def __init__(self,stackSize,numActions):
        super(DQN_image, self).__init__()
        self.conv1 = nn.Conv2d(stackSize,32,8,stride=4)
        self.conv2 = nn.Conv2d(32,64,4,stride=2)
        self.conv3 = nn.Conv2d(64,64,3,stride=2)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,numActions)

    def forward(self,x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = self.flatten(x)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x
