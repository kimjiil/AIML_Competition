import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import itertools
import cv2, PIL
import os, glob
import csv, platform
import torchvision

current_os = platform.system()
if current_os == "Linux":
    cfg = {
        'device': "cuda:5",
        "db_path": '/home/kji/workspace/jupyter_kji/samsumg_sem_dataset',
        'epochs': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'num_workers': 4,
        'n_fold': 5
    }
elif current_os == "Windows":
    cfg = {
        'device': "cuda:0",
        "db_path": 'D:/git_repos/samsung_sem',
        'epochs': 100,
        'batch_size': 4,
        'lr': 0.0002,
        'num_workers': 0,
        'n_fold': 5
    }

import wandb

wandb.login(key='0322000365224d30ef0694f60237c68767290052')
wandb.init(project="Samsung sem CycleGan", entity="kimjiil2013")

class CNN_classifier(nn.Module):
    def __init__(self):
        super(CNN_classifier, self).__init__()
        mobv3s = torchvision.models.mobilenet_v3_small(pretrained=True)
        feature = [nn.Sequential(nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2,2), padding=(1,1), bias=False),
                   nn.BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                   nn.Hardswish())]
        feature.extend([mobv3s.features._modules[module_key] for i, module_key in enumerate(mobv3s.features._modules.keys()) if i > 0])

        self.feature = nn.Sequential(*feature)
        self.avgpool = mobv3s.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=128, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=128, out_features=4, bias=True)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_img_list(abs_path):
    # abs_path = '/home/kji/workspace/jupyter_kji/samsumg_sem_dataset'

    # Dataset path
    sim_depth_path = os.path.join(abs_path, 'simulation_data/Depth')
    sim_sem_path = os.path.join(abs_path, 'simulation_data/SEM')

    train_path = os.path.join(abs_path, 'train')

    # only Test
    test_path = os.path.join(abs_path, 'test/SEM')

    sim_depth_img_path_dic = dict()
    for case in os.listdir(sim_depth_path):
        if not case in sim_depth_img_path_dic:
            sim_depth_img_path_dic[case] = []
        for folder in os.listdir(os.path.join(sim_depth_path, case)):
            img_list = glob.glob(os.path.join(sim_depth_path, case, folder, '*.png'))
            for img in img_list:
                sim_depth_img_path_dic[case].append(img)
                sim_depth_img_path_dic[case].append(img)

    sim_sem_img_path_dic = dict()
    for case in os.listdir(sim_sem_path):
        if not case in sim_sem_img_path_dic:
            sim_sem_img_path_dic[case] = []
        for folder in os.listdir(os.path.join(sim_sem_path, case)):
            img_list = glob.glob(os.path.join(sim_sem_path, case, folder, '*.png'))
            sim_sem_img_path_dic[case].extend(img_list)

    train_avg_depth = dict()
    with open(os.path.join(train_path, "average_depth.csv"), 'r') as csvfile:
        temp = csv.reader(csvfile)
        for idx, line in enumerate(temp):
            if idx > 0:
                depth_key, site_key = line[0].split('_site')
                depth_key = depth_key.replace("d", "D")
                site_key = "site" + site_key
                if not depth_key in train_avg_depth:
                    train_avg_depth[depth_key] = dict()

                train_avg_depth[depth_key][site_key] = float(line[1])

    train_img_path_dic = dict()
    for depth in os.listdir(os.path.join(train_path, "SEM")):
        if not depth in train_img_path_dic:
            train_img_path_dic[depth] = []
        for site in os.listdir(os.path.join(train_path, "SEM", depth)):
            img_list = glob.glob(os.path.join(train_path, "SEM", depth, site, "*.png"))
            train_img_path_dic[depth].extend([[temp_img, train_avg_depth[depth][site]] for temp_img in img_list])

    test_img_path_list = glob.glob(os.path.join(test_path, "*.png"))

    result_dic = dict()
    result_dic['sim'] = dict()
    result_dic['sim']['sem'] = sim_sem_img_path_dic
    result_dic['sim']['depth'] = sim_depth_img_path_dic
    result_dic['train'] = train_img_path_dic
    result_dic['test'] = np.array(test_img_path_list)
    result_dic['train_avg_depth'] = train_avg_depth

    return result_dic

result_dic = get_img_list(cfg['db_path'])

def split_dataset(data_dic, t_ratio):
    temp_dic = dict()
    for key in data_dic:
        temp_dic[key] = data_dic[key][:int(t_ratio*len(data_dic[key]))]
        data_dic[key] = data_dic[key][int(t_ratio*len(data_dic[key])):]

    return temp_dic, data_dic

train_dic, valid_dic = split_dataset(result_dic['train'], 0.8)

class cls_dataset(Dataset):
    def __init__(self, data_dic, transforms=None):
        super(cls_dataset, self).__init__()
        self.trasforms = transforms

        self._data = [[l[0], int(key.split('_')[-1]) % 100 / 10 - 1] for key in data_dic for l in data_dic[key]]

    def __getitem__(self, idx):
        img_path, label = self._data[idx]
        img = PIL.Image.open(img_path).convert("L")

        if self.trasforms:
            img = self.trasforms(img)

        img = np.array(img).astype(np.float32) / 255.
        if len(img.shape) == 2:
            img = img.reshape(1, *img.shape)

        return img, int(label)

    def __len__(self):
        return len(self._data)


horizon_transform = transforms.RandomHorizontalFlip(1.0)
rotate_transform = transforms.RandomRotation((180, 180))
vertical_transform = transforms.RandomVerticalFlip(1.0)

original_train_dataset = cls_dataset(train_dic)
original_valid_dataset = cls_dataset(valid_dic)

horizon_train_dataset = cls_dataset(train_dic, horizon_transform)
horizon_valid_dataset = cls_dataset(valid_dic, horizon_transform)

rotate_train_dataset = cls_dataset(train_dic, rotate_transform)
rotate_valid_dataset = cls_dataset(valid_dic, rotate_transform)

vertical_train_dataset = cls_dataset(train_dic, vertical_transform)
vertical_valid_dataset = cls_dataset(valid_dic, vertical_transform)

train_dataset = original_train_dataset + horizon_train_dataset + rotate_train_dataset + vertical_train_dataset
valid_dataset = original_valid_dataset + horizon_valid_dataset + rotate_valid_dataset + vertical_valid_dataset

train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True)

from tqdm.auto import tqdm

def valid(model, valid_dataloader, device):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for step_i, (img, label) in enumerate(tqdm(iter(valid_dataloader))):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            pred = model(img)
            pred_label = torch.argmax(pred, dim=1)
            correct += torch.sum(label == pred_label).item()
            total += label.shape[0]

    return correct / total

def Trainer(model, train_dataloader, valid_dataloader, device, epochs, checkpoint_path=None):
    best_epoch = 0
    best_accuracy = 0.0
    best_loss = 0
    if checkpoint_path:
        model = torch.load(checkpoint_path, map_location=device)

    loss_f = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    schedular = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = []
        for step_i, (img, label) in enumerate(tqdm(iter(train_dataloader))):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(img)
            loss = loss_f(pred, label)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

            wandb.log({
                'step_loss': loss.item()
            })
            if step_i == 0:
                break

        schedular.step()
        valid_acc = valid(model, valid_dataloader, device)

        wandb.log({
            'epoch_loss': np.mean(epoch_loss),
            'valid_acc': valid_acc
        })

        if valid_acc > best_accuracy:
            torch.save(model, './best_cnn_classifer.pth')
            best_accuracy = valid_acc
            best_loss = np.mean(epoch_loss)
            best_epoch = epoch
        print(f'epoch {epoch} / loss {np.mean(epoch_loss)} / acc {valid_acc}')
    print(f"bt epoch {best_epoch} / loss {best_loss} / best {best_accuracy}%")

model = CNN_classifier()
Trainer(model, train_dataloader, valid_dataloader, cfg['device'], cfg['epochs'])
