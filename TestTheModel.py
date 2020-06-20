from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import runpy
import pandas as pd
import numpy as np
import os
import string

class HARmodel(nn.Module):
    def __init__(self):
        super(HARmodel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 100, 2),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            # nn.Dropout(),
            nn.MaxPool1d(8),
            nn.Conv1d(100, 200, 1),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.MaxPool1d(8),
            nn.Conv1d(200, 100, 1),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            # nn.Dropout()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(100, 50, 2),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            # nn.Dropout(),
            nn.MaxPool1d(8))
        self.fc1 = nn.Linear(400, 100)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(100,6)

    def forward(self, x):
        # input.shape:(16,1,425)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # torch.Size([16, 400])
        # self.len_Linear=len(out)
        out = self.fc1(out)
        out=self.relu(out)
        out=self.fc2(out)
        return out
def restore():
    torch.load('Model/model6/classi_epoch_826.pt')
def restore_param():
    net1=HARmodel()

    return net1

model1=restore_param()
model1.eval()
model1.load_state_dict(torch.load('Model/model6/classi_epoch_826.pt'))

import csv
with open('Test319.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
# c=0
acc=0.
for row in rows[1:]:
    x=row[1:-1]
    label=int(row[-1])
    a=[]
    for i in x:
        i=float(i)
        a.append(i)

    x=np.array(a)
    # print(x)
    x=np.array([[x]])
    x=torch.from_numpy(x).float()
    y=model1(x)
    # pre=torch.max(y.data,1)
    # print(pre)
    # print(model1)
    # print(y)
    probability = torch.nn.functional.softmax(y,dim=1)#计算softmax，即该图片属于各类的概率
    """
     standing 0;walk 1;laying 2;run 3；down 4;up 5
    """
    max_value,index = torch.max(probability,1)
    if index==label:
        acc+=1
    index=index.numpy()#找到最大概率对应的索引号，该图片即为该索引号对应的类别
    if index==0:
        print("standing")
        # acc+=1
    elif index==1:
        print("walk")
        acc+=1
    elif index==2:
        print("laying")
        # acc+=1
    elif index==3:
        print("running")
        # acc+=1
    elif index==4:
        print("down")
    elif index==5:
        print("up")
acc/=len(rows)-1
print(acc*100,'%')
    # print(index[0])