from __future__ import print_function
import torch
import  torch.utils.tensorboard
from torch.autograd import Variable
from torch import nn
from torchviz import make_dot, make_dot_from_trace
from torch.utils.tensorboard import SummaryWriter

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import os


# from data import FeatureDataset
class FeatureDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.df = pd.read_csv(dir, header=None, sep=',')
        self.to_tensor = torch.FloatTensor()
        var_list = []
        # lendf=len(df[1])
        for i in range(1,2450):
            var_list.append(list(map(float, self.df.iloc[i, 3:564])))

        self.features = var_list

        # self.features=df.iloc[2:7353,0:562]
        # self.features=np.array(self.features).astype(np.float32)
        labels = list(map(int, self.df.iloc[1:2450, 564]))
        self.labels = np.array(labels)
        # self.root_dir=root_dir
        # self.labels=df.iloc[2:7353,564]

    def __len__(self):
        return len(self.df)-10

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        feature = self.features[idx]
        feature = np.array([feature])
        feature = feature.astype('float')
        # feature=np.tile(feature, (4, 1))
        # feature=feature.reshape(20,28)
        label = self.labels[idx]

        # feature=list(map(float,feature))

        # self.feature=feature.astype('float')

        # feature=self.to_tensor(feature)
        # label=self.labels[idx]
        sample = {'feature': feature, 'label': label}
        return sample


class Vali_FeatureDataset(Dataset):
    def __init__(self, vali_dir, transform=None):
        self.vali_df = pd.read_csv(vali_dir, header=None, sep=',')
        self.to_tensor = torch.FloatTensor()
        vali_var_list = []
        for i in range( 1,613):
            # Name: 562, Length: 2948, dtype: object
            vali_var_list.append(list(map(float, self.vali_df.iloc[i, 3:564])))

        self.vali_features = vali_var_list
        vali_labels = list(map(int, self.vali_df.iloc[1:613, 564]))
        self.vali_labels = np.array(vali_labels)

    def __len__(self):
        return len(self.vali_df)-10

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        vali_feature = self.vali_features[idx]
        vali_feature = np.array([vali_feature])
        vali_feature = vali_feature.astype('float')
        # vali_feature=np.tile(vali_feature, (4, 1))
        # feature=feature.reshape(20,28)
        vali_label = self.vali_labels[idx]

        # feature=list(map(float,feature))

        # self.feature=feature.astype('float')

        # feature=self.to_tensor(feature)
        # label=self.labels[idx]
        vali_sample = {'feature': vali_feature, 'label': vali_label}
        return vali_sample

class HARmodel(nn.Module):
    def __init__(self):
        super(HARmodel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, 5),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 32, 2),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(17664, 200)
        self.relu2=nn.ReLU()
        self.fc2=nn.Linear(200,4)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # torch.Size([16, 400])
        out = self.fc1(out)
        out=self.relu2(out)
        out=self.fc2(out)

        return out


def Train(train_loader, valid_loader, model, criterion, optimizer, device, save_model, epochs, save_dir,train_batch,test_batch):
    writer = SummaryWriter('D:\PytorchLog')
    model.train()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    train_losses = []
    print('-------------------start training--------------------------------------------------')
    for epoch in range(epochs):
        model.train()
        scheduler.step()
        acc_total_train=0
        # sum_loss=0.
        cnt=0
        train_loss_epoch=0.
        for i, sample in enumerate(train_loader):
            x = sample['feature']
            cnt+=1
            output = model(x.float())
            rightOrnot_train = torch.argmax(output, dim=1)
            # print(output.size())
            targets = sample['label']
            targets = targets.long()
            acc_train = 0.
            for j in range(len(rightOrnot_train)):
                res = rightOrnot_train[j] ==targets[j]
                acc_train += res
            acc_total_train += acc_train.item()
            loss = criterion(output, targets)
            writer.add_scalar('Train/Loss', loss.item(), epoch)

            train_loss_epoch+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            # loss.mean().backward()
            optimizer.step()
            if i % 100== 0:
                print('=====================================训练========================================')
                print("epoch:{}/{}|iter:{}|loss:{}".format(epoch, epochs, i, loss.item()))
        print("================================训练acc==================")
        acc_total_train/=cnt*train_batch
        train_loss_epoch/=cnt
        writer.add_scalar('Train/acc', acc_total_train, epoch)
        writer.add_scalar('Train/loss/mean/epoch', train_loss_epoch, epoch)
        # writer.flush()
        print('TrainAcc:',acc_total_train,'||TrainLoss:',train_loss_epoch)
        #######################vali################################################

        valid_mean_pts_loss = 0.0
        acc_total = 0.
        valid_loss = 0.
        model.eval()
        with torch.no_grad():
            valid_batch_cnt = 0.

            for valid_idx, valid_sample in enumerate(valid_loader):
                valid_batch_cnt += 1
                feature = valid_sample['feature']
                target_pts = valid_sample['label']
                target_pts = target_pts.long()
                output_pts = model(feature.float())
                rightOrnot=torch.argmax(output_pts,dim=1)

                # print('结果',rightOrnot,'标签',target_pts)
                acc = 0.
                for i in range(len(rightOrnot)):
                    res= rightOrnot[i]==target_pts[i]
                    acc+=res
                # res=rightOrnot == target_pts
                #     if i%30==0 and valid_idx%60==0:
                #         print('结果：',rightOrnot,rightOrnot.size(),'目标：',target_pts,target_pts.size())
                acc_total+=acc.item()
                # acc_total/=len(rightOrnot)
                valid_loss = criterion(output_pts, target_pts)
                valid_mean_pts_loss += valid_loss.item()
            valid_mean_pts_loss /= valid_batch_cnt * 1.0
            acc_total/=valid_batch_cnt*test_batch
            writer.add_scalar('Test/acc', acc_total, epoch)
            writer.add_scalar('Test/loss/mean/epoch', valid_mean_pts_loss, epoch)
            # writer.flush()

            print('=================================测试========================================================')
            print('valid:pts_loss:{:.6f}|acc:{}'.format(valid_mean_pts_loss,acc_total))
        if save_model:
            saved_model_name = os.path.join(save_dir, 'classi_epoch' + '_' + str(epoch) + '.pt')
            torch.save(model.state_dict(), saved_model_name)
    writer.close()
    return loss, 0.5
# draw my model
def vis():
    model=HARmodel()
    x = Variable(torch.randn(1,561).unsqueeze(1))
    y = model(x)
    vise_graph=make_dot(y, params=dict(model.named_parameters()))
    vise_graph.view()
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    model = HARmodel()
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer=torch.optim.RMSprop(model.parameters(),lr=0.001,alpha=0.9)
    # RMSProp相较于Adagrad的优点是在鞍点等地方，它在鞍点呆的越久，学习率会越大
    # optimizer=torch.optim.SGD(model.parameters(),lr=0.005,momentum=0.9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9,nesterov=False)
    train_dataset = FeatureDataset('D:\pycharm\HAR_CNN\data422\CleanData\Train422.csv')
    vali_dataset = Vali_FeatureDataset('D:\pycharm\HAR_CNN\data422\CleanData\Test422.csv')
    train_batch_in=8
    test_batch_in=32
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_in,
                                               num_workers=1,
                                               pin_memory=True,
                                               shuffle=True)
    vali_loader = torch.utils.data.DataLoader(vali_dataset,
                                              batch_size=test_batch_in,
                                              num_workers=1,
                                              pin_memory=True,
                                              shuffle=True)
    epochs = 15

    Train(train_loader, vali_loader, model, criterion, optimizer, device, True, epochs, 'D:\pycharm\HAR_CNN\Model\model24',train_batch_in,test_batch_in)

