import torch
from dataset_DM import TrainDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange, repeat
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torchvision import models

dataset = TrainDataset("/home/pod/shared-nvme/data/EyeOCT/train", width=224)
dataloader = DataLoader(dataset, num_workers=14, batch_size=16, shuffle=False)

resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet = resnet.cuda()
resnet.eval()

hints = []
for batch in tqdm(dataloader):
    img = batch['jpg']
    img = repeat(img, 'b h w n -> (b n) c h w', c=3)
    with torch.no_grad():
        feat = resnet(img.cuda()).cpu()
    hints.append(feat)

hints = torch.cat(hints, dim=0)
hints = rearrange(hints, '(b n) f -> b (n f)', n=6)
X = hints.numpy()
print(X.shape)

for cluster_nums in range(3, 50):
    print(f'cluster {cluster_nums}')
    kmeans = KMeans(n_clusters=cluster_nums, random_state=42)
    # 计算silhouette_score
    silhouette_avg = silhouette_score(X, kmeans.fit_predict(X))
    print(f'{cluster_nums} -- silhouette_score: {silhouette_avg}')


# print(kmeans.labels_, kmeans.cluster_centers_)




"""
conda activate 2024-eye
cd /home/pod/project/code/EyeOCT
python train_DM.py

nvitop
"""