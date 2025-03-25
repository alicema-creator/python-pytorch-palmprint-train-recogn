import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torchvision import models


import time
import sys
import os
import pickle
import numpy as np
from PIL import Image
import cv2


import matplotlib.pyplot as plt
plt.switch_backend('agg')


from models import MyDataset
from models import compnet
from utils import *



# path
train_set_file = './data/train.txt'
test_set_file = './data/test.txt'

path_rst = './rst'

#python_path = '/home/sunny/local/anaconda3/envs/torch37/bin/python'

# dataset
trainset = MyDataset(txt=train_set_file, transforms=None, train=True, imside=128, outchannels=1)
testset = MyDataset(txt=test_set_file, transforms=None, train=False, imside=128, outchannels=1)


#batch_size = 8
batch_size = 1


data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

data_loader_show = DataLoader(dataset=trainset, batch_size=1, shuffle=True)


if not os.path.exists(path_rst):
    os.makedirs(path_rst)


print('%s'%(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\ndevice-> ', device, '\n\n')


num_classes=600 # IITD: 460    KTU: 145    Tongji: 600    REST: 358    XJTU: 200

net = compnet(num_classes=num_classes)

# net.load_state_dict(torch.load('net_params.pkl'))

net.to(device)

#
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)  
scheduler = lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.8) 



