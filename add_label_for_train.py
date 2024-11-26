import pandas as pd

# 读取train.csv
train_df = pd.read_csv('/home/pod/shared-nvme/data/EyeOCT/train/train.csv')

# 读取inference.txt
with open('/home/pod/project/code/EyeOCT/OCT_Classification/inference.txt', 'r') as f:
    for i, line in enumerate(f):
        inference = line.strip().split(',')[:-1]
        assert len(inference) == len(train_df)
        inference = [int(x) for x in inference]
        train_df[f'infer_{i}'] = inference

    for i in range(len(train_df)):
        ls = [0,0,0,0]
        for j in range(6):
            cls = train_df[f'infer_{j}'].get(i)
            ls[cls] += 1

print(train_df)
train_df.to_csv('/home/pod/shared-nvme/data/EyeOCT/train/train_with_label.csv', index=False)