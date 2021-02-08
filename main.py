import os
from datetime import datetime


# 打印时间
def print_bar():
    nowtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "===========" * 4 + nowtime)


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets


class DataLoaderMaker:
    def __init__(self, path: str, p_train=0.8):
        """
        输入按文件夹分类好的数据集的路径与训练数据占比，构建DataLoader
        :param path: 数据集的路径
        :param p_train: 训练集占总训练数据的比例
        """
        self.path = path if os.path.exists(path) else None
        self.p_train = p_train

    def getDataLoader(self):
        if not self.path:
            return None
        transforms_train = transforms.Compose([transforms.ToTensor()])
        total_data = datasets.ImageFolder("./dataset",
                                          transform=transforms_train,
                                          target_transform=lambda t: torch.tensor([t]).float())
        n = len(total_data)
        ds_train, ds_valid = random_split(dataset=total_data, lengths=[int(0.8*n), n-int(0.8*n)])
        train = DataLoader(ds_train, batch_size=16, shuffle=True)
        valid = DataLoader(ds_valid, batch_size=16, shuffle=True)

        return train, valid, total_data

def trans_labels(labels, n):
    res = []
    for ele in labels:
        temp = [0] * n
        temp[int(ele.item())] = 1.
        res.append(temp)
    return torch.tensor(res)

DLM = DataLoaderMaker("./dataset")
dl_train, dl_valid, total_data = DLM.getDataLoader()

from matplotlib import pyplot as plt

plt.figure(figsize=(8,8))
for i in range(9):
    img, label = total_data[i]
    img = img.permute(1, 2, 0)
    ax = plt.subplot(3, 3, i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d" % label.item())
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

for x,y in dl_train:
    print(x.shape, y.shape)
    break

pool = nn.AdaptiveMaxPool2d((1,1))
t = torch.randn(10,8,32,32)
print(pool(t).shape)

class Net(nn.Module):

    def __init__(self, n=10):
        super(Net, self).__init__()
        self.out_n = n
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, n)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y

net = Net()
print(net)

import torchkeras
print(torchkeras.summary(net, input_shape=(3, 28, 28)))

import pandas as pd
from sklearn.metrics import roc_auc_score

model = net
model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.loss_func = nn.BCELoss()
model.metric_func = lambda y_pred, y_true: roc_auc_score(y_true.data.numpy(), y_pred.data.numpy())
model_metric_name = "auc"

def train_step(model, features, labels):
    labels_tensor = trans_labels(labels, model.out_n)
    model.train()
    model.optimizer.zero_grad()
    predictions = model(features)
    loss = model.loss_func(predictions, labels_tensor)
    metric = model.metric_func(predictions, labels_tensor)
    loss.backward()
    model.optimizer.step()

    return loss.item(), metric.item()

def valid_step(model, features, labels):
    labels_tensor = trans_labels(labels, model.out_n)
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        loss = model.loss_func(predictions, labels_tensor)
        metric = model.metric_func(predictions, labels_tensor)

    return loss.item(), metric.item()

features, labels = next(iter(dl_train))
print(train_step(model, features, labels))