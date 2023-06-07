import os, copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal
import random
import math
from data_process import file2list, name2index,resample,file2index
class ECGDatasetForPreTrain(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, data_path):
        self.data = file2list(data_path)
    def __getitem__(self, index):
        fid = self.data[index % len(self.data)].strip()
        file_path = os.path.join(config.data_dir,fid)
        x = np.load(file_path)
        x = x[[1],:]
        x = x.astype(np.float32)
        if config.data_standardization:
            x = (x - np.mean(x)) / np.std(x)
        x = resample(x[0],len(x[0]) * config.target_sample_rate // config.data_sample_rate)

        x = torch.tensor(x, dtype=torch.float32)
        return x.unsqueeze(0)

    def __len__(self):
        return len(self.data)
class segment_dataset2(Dataset):
    def __init__(self,data_path):
        self.data = file2list(data_path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        fid = self.data[index % len(self.data)].strip()
        file_path = os.path.join(config.segment_data_dir,fid)
        seg = np.load(file_path)
        seg = seg.astype(np.float32)
        if config.data_standardization:
            ecg_II = (seg[0]-np.mean(seg[0]))/np.std(seg[0])
        else:
            ecg_II = seg[0]
        mask_arr = seg[1:,:]
        x = torch.tensor(ecg_II, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(mask_arr,dtype=torch.float32)
        return x,y
class segment_dataset(Dataset):
    def __init__(self,file_dir):
        self.filelist = os.listdir(file_dir)
        self.file_dir = file_dir
    def __len__(self):
        return len(self.filelist)
    def __getitem__(self,item):
        # try:
        #     seg = np.load(os.path.join(self.file_dir,self.filelist[item]),type=np.float32)
        #     ecg_II = (seg[0]-np.mean(seg[0]))/np.std(seg[0])
        #     mask_arr = seg[1:,:]
        # except:
        #     print(self.filelist[item])
        seg = np.load(os.path.join(self.file_dir,self.filelist[item]))
        seg = seg.astype(np.float32)
        if config.data_standardization:
            ecg_II = (seg[0]-np.mean(seg[0]))/np.std(seg[0])
        else:
            ecg_II = seg[0]
        mask_arr = seg[1:,:]
        x = torch.tensor(ecg_II, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(mask_arr,dtype=torch.float32)
        return x,y
class classifier_dataset(Dataset):
    def __init__(self,label_path,data_path):
        name2idx = name2index(config.classifier_arraythmia)
        self.file2idx = file2index(label_path,name2idx)
        self.filelist = file2list(data_path)
    def __len__(self):
        return len(self.filelist)
    def __getitem__(self,item):
        fid = self.filelist[item]
        x = np.load(os.path.join(config.classifier_data_dir,fid))
        x = x.astype(np.float32)
        if config.data_standardization:
            x = (x-np.mean(x))/np.std(x)
        
        if config.classifier_loss=='BCE':
            target = np.zeros(config.classifier_class_n)
            target[self.file2idx[fid]]=1
        else:
            target = self.file2idx[fid]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(target,dtype=torch.long)
        return x,target
class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """
    
    def __init__(self, data_path, train=True, two_crop_transform=False):
        super(ECGDataset, self).__init__()
        dd = torch.load(data_path)
        self.train = train
        self.data = dd['train'] if train else dd['val']
        self.idx2name = dd['idx2name']
        self.file2idx = dd['file2idx']
        self.wc = 1. / np.log(dd['wc'])
        if 'data_standardization' in dir(config) and config.data_standardization:
            self.train_mean = dd['train_mean']
            self.train_std = dd['train_std']
            self.train_mean = self.train_mean[[0, 1, 6, 7, 8, 9, 10, 11]]
            self.train_std = self.train_std[[0, 1, 6, 7, 8, 9, 10, 11]]
            self.train_mean = self.train_mean * config.inputUnit_uv / config.targetUnit_uv  # 符值转换
            self.train_std = self.train_std * config.inputUnit_uv / config.targetUnit_uv  # 符值转换

    def __getitem__(self, index):
        fid = self.data[index % len(self.data)]
        file_path = os.path.join(config.train_dir, fid+'.npy')
        x = np.load(file_path)
        x = x[[0, 1, 6, 7, 8, 9, 10, 11], :]  # 截出8导联
        x = x.astype(np.float32)
        x = x * config.inputUnit_uv / config.targetUnit_uv  # 符值转换
        # x = x.T  # 注意这次转置
        if 'data_standardization' in dir(config) and config.data_standardization:
            x = (x - self.train_mean) / self.train_std
        target = None
        # 为单标签多分类提供标量的loss，0 <= target < num_classes
        if 'loss_function' in dir(config) and (
                config.loss_function == 'WeightedCrossEntropyLoss' or config.loss_function == 'MultiClassFocalLoss'):
            target = self.file2idx[fid][0]
            target = torch.tensor(target, dtype=torch.long)
        else:
            target = np.zeros(config.num_classes)
            target[self.file2idx[fid]] = 1
            target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

        if 'loss_function' in dir(config) and config.loss_function == 'WeightSinglelabel':
            weight = -1
            for i in self.file2idx_w[fid]:
                if weight < self.class_weight_dict[i]:
                    weight = self.class_weight_dict[i]  # 取多标签中最大权重为当前权重
            weight = weight + 1
            target = np.append(target, weight)
            targetwithw = torch.tensor(target, dtype=torch.float32)
            return x, targetwithw
        elif self.train and 'loss_function' in dir(config) and config.loss_function == 'MLB_SupConLoss':
            cur_error_rate = np.ones(1, dtype=np.float32)
            # 处理标签为空即'其他'的情况
            if len(self.file2idx_full[fid]) == 0:
                cur_error_rate[0] = self.error_rate_list[-1]
            else:
                cur_error_rate_list = []
                for idx in self.file2idx_full[fid]:
                    cur_error_rate_list.append(self.error_rate_list[idx])

                if config.MLB_multi_label_error_rate_mode == 'aver':
                    cur_error_rate[0] = sum(cur_error_rate_list) / len(self.file2idx_full[fid])
                elif config.MLB_multi_label_error_rate_mode == 'max':
                    cur_error_rate[0] = max(cur_error_rate_list)

            target = torch.tensor(target, dtype=torch.float32)
            cur_error_rate = torch.tensor(cur_error_rate, dtype=torch.float32)
            return x, [target, cur_error_rate]

        return x, target

    def __len__(self):
        if 'twice_data_augmentation' in dir(config) and config.twice_data_augmentation:
            return len(self.data) * 2
        return len(self.data)
