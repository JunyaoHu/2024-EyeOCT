import argparse
import numpy as np
import pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from utils import PDBLOCTDataSet, OCT_for_PDBLOCTDataSet
from models import resnet50, PDBLNet
from sklearn.metrics import accuracy_score, f1_score

class Solver :
    def __init__(self, args):
        self.args = args

    def create_model(self, model_name, n_classes):
        if model_name == 'resnet' :
            model = resnet50(pretrained = True)
            model.fc = nn.Linear(512 * model.block.expansion, n_classes)
        else :
            raise ValueError('model should be resnet!')

        return model

    def save_PDBL(self, PDBL, fname):
        with open(Path(self.args.save_dir).joinpath(fname).__str__(), 'wb') as fp :
            pickle.dump(PDBL, fp)

    def load_PDBL(self, fname):
        with open(Path(self.args.save_dir).joinpath(fname).__str__(), 'rb') as fp :
            PDBL = pickle.load(fp)

        return PDBL

    def save_model(self, model, fname):
        torch.save(model.state_dict(), Path(self.args.save_dir).joinpath(fname).__str__())

    def load_model(self, model, fname):
        ckpt = torch.load(Path(self.args.save_dir).joinpath(fname).__str__(), map_location = self.args.device)
        model.load_state_dict(ckpt)

        return model

    def train_model(self, model, train_loader):
        model.train()
        # total_params = sum(p.numel() for p in model.parameters())
        # print('原总参数个数:{}'.format(total_params))
        # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print('需训练参数个数:{}'.format(total_trainable_params))

        optimizer = optim.SGD(model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay, momentum = 0.9)
        criterion = nn.CrossEntropyLoss()
        losses, acces = [], []

        for epoch in trange(self.args.max_epoch) :
            train_loss = 0.0
            train_acc = 0.0
            for img, label in tqdm(train_loader):
                torch.manual_seed(0)
                torch.cuda.manual_seed(0)

                img, label = img.to(self.args.device), label.to(self.args.device)
                _, output = model(img)

                _, pred = torch.max(output, dim = 1)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                num_correct = (pred == label).sum().item()
                acc = num_correct / len(label)
                train_loss += loss.item()
                train_acc += acc

            print(f'[{epoch + 1:02d}/{self.args.max_epoch:02d}], train_loss:{train_loss / len(train_loader):.5f}, '
                  f'train_acc:{train_acc / len(train_loader):.5f}')

            losses.append(round(train_loss / len(train_loader), 5))
            acces.append(round(train_acc / len(train_loader), 5))

        return model

    def valid_model(self, model, val_loader):
        model.eval()
        model_true = []
        model_pred = []
        with torch.no_grad() :
            for img, label in val_loader:
                torch.manual_seed(0)
                torch.cuda.manual_seed(0)

                img, label = img.to(self.args.device), label.to(self.args.device)
                _, output = model(img)
                _, pred = torch.max(output, dim = 1)

                num_correct = (pred == label).sum().item()
                acc = num_correct / len(label)

                model_true.extend(label.detach().cpu().numpy())
                model_pred.extend(pred.detach().cpu().numpy())

        model_acc = accuracy_score(model_true, model_pred)
        model_f1 = f1_score(model_true, model_pred, average = 'macro')

        print(f'Accuracy of {self.args.model_name}: ', model_acc)
        print(f'F1 score of {self.args.model_name}: ', model_f1)

    def inference_model(self, model, val_loader):
        model.eval()
        # model_true = []
        model_pred = []
        with torch.no_grad() :
            for img, label in tqdm(val_loader):
                torch.manual_seed(0)
                torch.cuda.manual_seed(0)

                img, _ = img.to(self.args.device), label.to(self.args.device)
                _, output = model(img)
                _, pred = torch.max(output, dim = 1)

                model_pred.extend(pred.detach().cpu().numpy())
                # 写入一个txt文件，一行一个数字

        with open(f'inference.txt', 'a') as f:
            for item in model_pred:
                f.write(str(item) + ',')
            f.write('\n')

        print("done")

    def run(self) :
        train_dataset = PDBLOCTDataSet(data_path = self.args.train_dir, n_classes = self.args.n_classes, train_model = True)
        # val_dataset = PDBLOCTDataSet(data_path = self.args.val_dir, n_classes = self.args.n_classes, train_model = True)
        val_dataset = OCT_for_PDBLOCTDataSet(data_path = self.args.val_dir, n_classes = self.args.n_classes, image_index=self.args.image_index)
        train_loader = DataLoader(train_dataset, batch_size = self.args.batch_size, num_workers=14, shuffle = self.args.train_model)
        val_loader = DataLoader(val_dataset, batch_size = self.args.batch_size, num_workers=14, shuffle = False)

        self.args.n_item_train = len(train_dataset)
        self.args.n_item_val = len(val_dataset)

        self.args.model_name = 'resnet'
        self.args.n_features = 3840 * 3
        print('<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('This model is', self.args.model_name)

        # scale = 0.001
        # n_components = min(int(self.args.n_item_train * 0.9), 2000)
        # PDBL = PDBLNet(isPCA = True, n_components = n_components, scale = scale)
        model = self.create_model(model_name = self.args.model_name, n_classes = self.args.n_classes)
        model = model.to(self.args.device)

        if not Path(self.args.save_dir).exists() :
            Path(self.args.save_dir).mkdir(exist_ok = True, parents = True)

        if self.args.train_model:
            model = self.train_model(model, train_loader)
            # self.valid_model(model, val_loader)
            fname = f'oct2017_classification_finetune_resnet_10epochs.pth'
            self.save_model(model, fname)
        else:
            fname = f'oct2017_classification_finetune_resnet_10epochs.pth'
            model = self.load_model(model, fname)
            # self.valid_model(model, val_loader)
            self.inference_model(model, val_loader)

        # fname = f'PDBL_with_finetune_on_resnet.pkl'
        # PDBL = self.train_PDBL(model, train_loader, PDBL)
        # PDBL = self.load_PDBL(fname)
        # self.valid_PDBL(model, val_loader, PDBL)

        # fname = f'PDBL_with_finetune_on_resnet.pkl'
        # self.save_PDBL(PDBL, fname)

        # PDBL_reload = self.load_PDBL(fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type = str, default = "cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--save_dir', type = str, default = 'checkpoint')
    parser.add_argument('--train_dir', type = str, default = 'OCT_Dataset/Processed_OCT2017/train')
    parser.add_argument('--val_dir', type = str, default = 'OCT_Dataset/Processed_OCT2017/test')
    parser.add_argument('--n_classes', type=int, default = 4)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--max_epoch', type = int, default = 10)
    parser.add_argument('--lr', type = float, default = 5e-3)
    parser.add_argument('--weight_decay', type = float, default = 1e-4)
    # parser.add_argument('--train_model', type = bool, default = False)
    parser.add_argument("--train_model", action='store_true')
    parser.add_argument('--image_index', type = int, default = 0)
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()


"""
python trainer.py \
    --save_dir /home/pod/project/code/EyeOCT/pretrained/OCT_Classification \
    --train_dir /home/pod/shared-nvme/data/OCT2017/OCT2017/train \
    --val_dir /home/pod/shared-nvme/data/OCT2017/OCT2017/test\
    --n_classes 4 \
    --train_model

CUDA_VISIBLE_DEVICES=1 \
python trainer.py \
    --save_dir /home/pod/project/code/EyeOCT/pretrained/OCT_Classification \
    --train_dir /home/pod/shared-nvme/data/OCT2017/OCT2017/train \
    --val_dir /home/pod/shared-nvme/data/EyeOCT/train \
    --n_classes 4 \
    --image_index 0
"""