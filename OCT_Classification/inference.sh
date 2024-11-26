#!/bin/bash

for i in {0..5}
do
    python trainer.py \
        --save_dir /home/pod/project/code/EyeOCT/pretrained/OCT_Classification \
        --train_dir /home/pod/shared-nvme/data/OCT2017/OCT2017/train \
        --val_dir /home/pod/shared-nvme/data/EyeOCT/train \
        --n_classes 4 \
        --image_index $i
done
