from torch.utils.data import DataLoader
from dataset_DM import TrainDatasetWithLabel

from einops import rearrange
from pathlib import Path
import torch.nn as nn
import torch
from torchvision import models
import torch.optim as optim
from tqdm import tqdm, trange
import os
import torch.nn.functional as F

from cldm.classifier import Classifier

batch_size = 128
max_epoch = 50
lr = 5e-4

save_dir = "/home/pod/shared-nvme/tensorboard/logs/CF_classfier"

dataset = TrainDatasetWithLabel("/home/pod/shared-nvme/data/EyeOCT/train")
dataloader = DataLoader(dataset, num_workers=14, batch_size=batch_size, shuffle=True)

print('<<<<<<<<<<<<<<<<<<<<<<<<< train >>>>>>>>>>>>>>>>>>>>>>>>>')

resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for param in resnet.parameters():
    param.requires_grad = False

resnet.cuda()

model = Classifier(4)
model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
losses, acces = [], []

def save_model(model, fname):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), Path(save_dir).joinpath(fname).__str__())

def load_model(model, fname):
    ckpt = torch.load(Path(save_dir).joinpath(fname).__str__(), map_location = "cuda")
    model.load_state_dict(ckpt)

torch.manual_seed(0)
torch.cuda.manual_seed(0)
    
for epoch in range(max_epoch):

    train_loss = 0.0
    train_acc = 0.0

    for batch in tqdm(dataloader):

        img   = batch['hint'].cuda()
        label = batch['label'].cuda()
        img   = rearrange(img, 'b h w c -> b c h w')
        with torch.no_grad():
            img = resnet(img)
        output = model(img)

        _, pred = torch.max(output, dim=1)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_correct = (pred == label).sum().item()
        acc = num_correct / len(label)
        train_loss += loss.item()
        train_acc += acc
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    print(f'[{epoch + 1:02d}], train_loss:{train_loss:.5f}, '
            f'train_acc:{train_acc:.5f}')
    fname = f'classification_e{epoch}_acc{train_acc:.4}.pth'
    save_model(model, fname)


"""
conda activate 2024-eye
cd /home/pod/project/code/EyeOCT
python train_CF_classifier.py

nvitop
"""