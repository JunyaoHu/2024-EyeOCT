# from share import *

# import pytorch_lightning as pl
# from torch.utils.data import DataLoader
# from dataset_DM import ValidDataset
# from cldm.model import create_model, load_state_dict

# # Configs
# resume_path = None
# batch_size = 2
# learning_rate = 4e-5
# accumulate = 4

# # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
# model = create_model('./models/classfier.yaml').cpu()
# if resume_path:
#     model.load_state_dict(load_state_dict(resume_path, location='cpu'))
# model.learning_rate = learning_rate

# save_dir = "/home/pod/shared-nvme/tensorboard/logs/CF_classfier"

# # Misc
# dataset = ValidDataset("/home/pod/shared-nvme/data/EyeOCT/valid")
# dataloader = DataLoader(dataset, num_workers=14, batch_size=batch_size, shuffle=True)
# trainer = pl.Trainer(strategy='ddp', gpus=2, precision=32, default_root_dir=save_dir, accumulate_grad_batches=accumulate, max_steps=-1)
# # trainer = pl.Trainer(gpus=1, callbacks=[logger])
# trainer.fit(model, dataloader)

# """
# conda activate 2024-eye
# cd /home/pod/project/code/EyeOCT
# python train_DM.py

# nvitop
# """