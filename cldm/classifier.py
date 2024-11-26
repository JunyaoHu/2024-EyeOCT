import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(self, n_classes, ckpt_path=None, ignore_keys=[]):
        super().__init__()
        self.proj_in  = nn.Linear(1000, 512)
        self.proj_mid  = nn.Linear(512, 64)
        self.proj_out = nn.Linear(64, n_classes)

        # 导入预训练模型
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.proj_mid(x)
        x = self.proj_out(x)
        return x
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")