import os
import sys
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import glob2
import random
from torchvision import transforms
import numpy as np
from torch.utils.data import ConcatDataset,Subset
random.seed(1)

with open('utils/data_Txt/IM_train.txt') as f:
    lines = f.readlines()
    IM_train_targets = [int(line.strip().split(',')[1]) for line in lines]
with open('utils/data_Txt/IM_test.txt') as f:
    lines = f.readlines()
    IM_test_targets = [int(line.strip().split(',')[1]) for line in lines]
with open('utils/data_Txt/GA_train.txt') as f:
    lines = f.readlines()
    GA_train_targets = [int(line.strip().split(',')[1]) for line in lines]
with open('utils/data_Txt/GA_test.txt') as f:
    lines = f.readlines()
    GA_test_targets = [int(line.strip().split(',')[1]) for line in lines]

def default_loader(path):
    try:   
        img = Image.open(path)
        return img
    except:
        print("Cannot read image: {}".format(path))

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomAffine(degrees=0,translate=(0.2,0.2)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=30),
        # transforms.RandomAffine(degrees=0,scale=(0.8,1.2)),
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
class CovidDataset(Dataset):
    def __init__(self, filename='../data/covid19/', data_trans=None):
        self.data = np.load(filename+'xdata.npy')
        self.targets = np.load(filename+'ydata.npy')
        self.targets = np.squeeze(self.targets)
        self.transform = data_trans
        self.data = torch.Tensor(self.data)
        self.data = torch.einsum('bxyz->bzxy', self.data)
    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        return img, label
class customData(Dataset):# 
    def __init__(self, txt_path, dataset = '', data_trans=None,loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [line.strip().split(',')[0] for line in lines]
            self.img_label = [int(line.strip().split(',')[1]) for line in lines]
        self.trans = data_trans
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)
        img = self.trans(img)
        return img, label

IM_image_datasets = {
    x: customData(txt_path=('utils/data_Txt/'+'IM_'+x+'.txt'),
        data_trans = data_transforms['train'],
        dataset= x ) for x in ['train','test','val']}
GA_image_datasets = {
    x: customData(txt_path=('utils/data_Txt/'+'GA_'+x+'.txt'),
        data_trans = data_transforms['train'],
        dataset= x ) for x in ['train','test','val']}

Covid_image = CovidDataset(filename='/home/FedTC/data/covid19/')
train_covid = Subset(Covid_image,range(int(1281*0.7)))+Subset(Covid_image,range(1281,1281+int(3001*0.7)))+Subset(Covid_image,range(4282,4282+int(1656*0.7)))+Subset(Covid_image,range(5938,5938+int(3270*0.7)))
Covid_targets = [i[1] for i in train_covid]
test_covid = Subset(Covid_image,range(int(1281*0.7),int(1281*0.9)))+Subset(Covid_image,range(1281+int(3001*0.7),1281+int(3001*0.9)))+Subset(Covid_image,range(4282+int(1656*0.7),4282+int(1656*0.9)))+Subset(Covid_image,range(5938+int(3270*0.7),5938+int(3270*0.9)))
val_covid = Subset(Covid_image,range(int(1281*0.9),1281))+Subset(Covid_image,range(1281+int(3001*0.9),4282))+Subset(Covid_image,range(4282+int(1656*0.9),5938))+Subset(Covid_image,range(5938+int(3270*0.9),len(Covid_image)))