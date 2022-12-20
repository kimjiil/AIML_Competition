import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import itertools
import cv2, PIL
import os, glob
import csv, platform

current_os = platform.system()
if current_os == "Linux":
    _path = '/home/kji/workspace/jupyter_kji/samsumg_sem_dataset'
    cfg = {
        'device': "cuda:5",
        "db_path": _path,
        'epochs': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'num_workers': 4,
        'n_fold': 5
    }
elif current_os == "Windows":
    _path = 'D:/git_repos/samsung_sem'
    cfg = {
        'device': "cuda:0",
        "db_path": _path,
        'epochs': 100,
        'batch_size': 4,
        'lr': 0.0002,
        'num_workers': 0,
        'n_fold': 5
    }


'''
    cnn classifier를 사용해서 case를 분류함
    Generator를 4가지 case를 나눠서 학습함
'''

