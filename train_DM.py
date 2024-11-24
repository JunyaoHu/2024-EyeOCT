from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset_DM import TrainDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Configs
resume_path = None
batch_size = 1
learning_rate = 2e-5
ddim_steps = 10
accumulate = 8

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
if resume_path:
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate

save_dir = "/home/pod/shared-nvme/tensorboard/logs/OCT_DM"

# Misc
dataset = TrainDataset("/home/pod/shared-nvme/data/EyeOCT/train")
dataloader = DataLoader(dataset, num_workers=14, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=250, max_images=8, increase_log_steps=True, log_images_kwargs=dict(ddim_steps=ddim_steps))
trainer = pl.Trainer(strategy='ddp', gpus=2, precision=32, callbacks=[logger], default_root_dir=save_dir, accumulate_grad_batches=accumulate)
# trainer = pl.Trainer(gpus=1, callbacks=[logger])
trainer.fit(model, dataloader)

"""
conda activate 2024-eye
cd /home/pod/project/code/EyeOCT
python train_DM.py

nvitop
"""