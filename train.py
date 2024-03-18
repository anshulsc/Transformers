# Cats and dogs 


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

from linformer import Linformer

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

from vit import ViT
# from Vit_torch import ViT, ModelArgs

import glob
from itertools import chain
import os 
import random
import zipfile 

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Training Settings
batch_size = 64
epochs = 1
lr = 3e-5
gamma = 0.7
seed = 42

# Seed everything for reproducibility
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)


print(f"{os.getcwd()}")
os.makedirs('data', exist_ok=True)
train_dir = 'data/train'
test_dir = 'data/test'

# # Unzip the data
# with zipfile.ZipFile('data/train.zip') as train_zip:
#     train_zip.extractall('data')

# with zipfile.ZipFile('data/test.zip') as test_zip:
#     test_zip.extractall('data')


train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))



print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")

labels = [path.split('/')[-1].split('.')[0] for path in train_list]

train_list, val_list = train_test_split(train_list, test_size=0.2, 
                                        stratify=labels,
                                        random_state=seed)

print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(val_list)}")



train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label

efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

train_data = CatsDogsDataset(train_list, transform=train_transforms)
valid_data = CatsDogsDataset(val_list, transform=test_transforms)
test_data = CatsDogsDataset(test_list, transform=test_transforms)


train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

# args  = ModelArgs(img_size=224, patch_size=32, out_dim=2, channels=3)

model = ViT(
    image_size = 224,
    patch_size = 32,
    num_classes = 2,
    dim = 128,
    transformer = efficient_transformer
).to(device)

model.to(device)

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}")
              
    with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

