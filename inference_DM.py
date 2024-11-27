from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset_DM import ValidDataset
from cldm.model import create_model, load_state_dict

# Configs
resume_path = "/home/pod/shared-nvme/tensorboard/logs/OCT_DM/lightning_logs/DM_global512_cls256/checkpoints/epoch=175-step=16543.ckpt"
batch_size = 8

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
if resume_path:
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))

save_dir = "/home/pod/shared-nvme/tensorboard/logs/OCT_DM"

# Misc
dataset = ValidDataset("/home/pod/shared-nvme/data/EyeOCT/val")
dataloader = DataLoader(dataset, num_workers=14, batch_size=batch_size, shuffle=False)

trainer = pl.Trainer(strategy='ddp', gpus=2, precision=32, default_root_dir=save_dir)
trainer.test(model, dataloader)

"""
conda activate 2024-eye
cd /home/pod/project/code/EyeOCT
python inference_DM.py

nvitop

cd /home/pod/project/code/EyeOCT/log_valid/1124-1830/samples
zip submit.zip -r ./
"""