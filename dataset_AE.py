import cv2
import numpy as np
import os
import pandas as pd

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, path, width=256):
        self.path = path
        self.data = pd.read_csv(os.path.join(self.path, 'train.csv'))
        self.width = width

    def __len__(self):
        return len(self.data)
        # return 64

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        PID = item['PID']
        CF_path = item['CF_path']
        OCT_path = item['OCT_path']
        prompt = "A cross-section of the retina obtained using optical coherence tomography, medical image, black and white"

        source = cv2.imread(os.path.join(self.path, 'CF', CF_path))
        h, w, _ = source.shape
        source = source[:, w//2-h//2:w//2+h//2, :]
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source = cv2.resize(source, (self.width, self.width))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

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
            txt=prompt, 
            hint=source, 
            id=idx,
            PID=PID,
            CF_path=CF_path,
            OCT_path=OCT_path
        )
    
class ValidDataset(Dataset):
    def __init__(self, path, width=256):
        self.path = path
        self.data = pd.read_csv(os.path.join(self.path, 'val.csv'))
        self.width = width

    def __len__(self):
        return len(self.data)
        # return 8

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        PID = item['PID']
        CF_path = item['CF_path']
        prompt = "A cross-section of the retina obtained using optical coherence tomography, medical image, black and white"

        source = cv2.imread(os.path.join(self.path, 'CF', CF_path))
        h, w, _ = source.shape
        source = source[:, w//2-h//2:w//2+h//2, :]
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source = cv2.resize(source, (self.width, self.width))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = np.zeros((self.width, self.width, 6))

        return dict(
            jpg=target, 
            txt=prompt, 
            hint=source, 
            id=idx,
            PID=PID,
            CF_path=CF_path,
        )

if __name__ == '__main__':

    data_path = "/home/pod/shared-nvme/data/EyeOCT/train"
    train_dataset = TrainDataset(data_path)
    print(len(train_dataset)) # 750

    # for i in range(750):
    #     item = train_dataset[i]
    #     OCT_path = item['OCT_path']
    #     print(i, OCT_path)
    
    item = train_dataset[0]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    id_ = item['id']
    PID = item['PID']
    CF_path = item['CF_path']
    OCT_path = item['OCT_path']
    print(txt)
    print(jpg.shape)
    print(hint.shape)
    print(id_)
    print(PID)
    print(CF_path)
    print(OCT_path)

    data_path = "/home/pod/shared-nvme/data/EyeOCT/val"
    valid_dataset = ValidDataset(data_path)
    print(len(valid_dataset))

    item = valid_dataset[0]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    id_ = item['id']
    PID = item['PID']
    CF_path = item['CF_path']
    print(txt)
    print(jpg.shape)
    print(hint.shape)
    print(id_)
    print(PID)
    print(CF_path)