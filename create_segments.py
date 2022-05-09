import numpy as np
import pandas as pd
import seaborn as sns
import torch
import glob
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler


#Transforms
transformer=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])

data_path='/home/christos_sevastopoulos/Desktop/toy_dataset_stanford/Training'

data_count=len(glob.glob(data_path+'/**/*.jpg'))
print(data_count)

########## random_split ################

data_loader=DataLoader(
    torchvision.datasets.ImageFolder(data_path,transform=transformer),
)

train_dataset, test_dataset = random_split(data_loader, (1300, 17))

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)




############## SubsetRandomSampler ##########################33



dataset_size = len(data_loader)
dataset_indices = list(range(dataset_size))

np.random.shuffle(dataset_indices)
test_split_index = int(np.floor(0.2 * dataset_size))

train_idx, test_idx = dataset_indices[test_split_index:], dataset_indices[:test_split_index]

train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = DataLoader(dataset=data_loader, shuffle=False, batch_size=1, sampler=train_sampler)
test_loader = DataLoader(dataset=data_loader, shuffle=False, batch_size=1, sampler=test_sampler)
