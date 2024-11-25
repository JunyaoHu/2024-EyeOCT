import cv2
import numpy as np
import os
import pandas as pd
from einops import rearrange
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, path, width=512, dd=64):
        self.path = path
        self.data = pd.read_csv(os.path.join(self.path, 'train.csv'))
        self.width = width
        self.dd = dd

    def __len__(self):
        return len(self.data)
        # return 64

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        PID = item['PID']
        CF_path = item['CF_path']
        OCT_path = item['OCT_path']

        source = cv2.imread(os.path.join(self.path, 'CF', CF_path))
        h, w, _ = source.shape
        source = source[:, w//2-h//2:w//2+h//2, :]
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source = cv2.resize(source, (self.width, self.width))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        global_hint = source

        target = np.zeros((self.width, self.width, 6))
        for i in range(6):
            target_i = cv2.imread(os.path.join(self.path, 'OCT', OCT_path, f"{OCT_path}_{i}.jpg"), cv2.IMREAD_GRAYSCALE)
            if idx <= 728:
                target_i = target_i[:496,-768:]
            target_i = cv2.resize(target_i, (self.width, self.width))
            target[:, :, i] = target_i
        
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(
            jpg=target, 
            txt="", 
            hint=global_hint, 
            id=idx,
            PID=PID,
            CF_path=CF_path,
            OCT_path=OCT_path
        )
    
class ValidDataset(Dataset):
    def __init__(self, path, width=512, dd=64):
        self.path = path
        self.data = pd.read_csv(os.path.join(self.path, 'val.csv'))
        self.width = width
        self.dd = dd

    def __len__(self):
        return len(self.data)
        # return 8

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        PID = item['PID']
        CF_path = item['CF_path']

        source = cv2.imread(os.path.join(self.path, 'CF', CF_path))
        h, w, _ = source.shape
        source = source[:, w//2-h//2:w//2+h//2, :]
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source = cv2.resize(source, (self.width, self.width))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        global_hint = source

        # Normalize target images to [-1, 1].
        target = np.zeros((self.width, self.width, 6))

        return dict(
            jpg=target, 
            txt="", 
            hint=global_hint, 
            id=idx,
            PID=PID,
            CF_path=CF_path,
        )

if __name__ == '__main__':

    data_path = "/home/pod/shared-nvme/data/EyeOCT/train"
    train_dataset = TrainDataset(data_path)
    print(len(train_dataset)) # 750

    item = train_dataset[0]
    txt = item['txt']
    jpg = item['jpg']
    # hint_global = item['hint'][0]
    # hint_local = item['hint'][1]
    id_ = item['id']
    PID = item['PID']
    CF_path = item['CF_path']
    OCT_path = item['OCT_path']
    print(txt)
    print(jpg.shape)
    # print(hint_global.shape)
    # print(hint_local.shape)
    print(id_)
    print(PID)
    print(CF_path)
    print(OCT_path)