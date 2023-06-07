# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 18:44
数据预处理：
    1.构建label2index和index2label
    2.划分数据集
@ author: javis
'''
import os, torch
import numpy as np
from config import config
from xml.dom import minidom
from scipy import signal
# 保证每次划分数据一致
np.random.seed(41)

def file2index(path, name2idx):
    '''
    获取文件id对应的标签类别
    :param path:文件路径
    :return:文件id对应label列表的字段
    '''

    file2index = dict()
    for line in open(path, encoding='utf-8'):
        if len(line.strip()) == 0:
            continue
        arr = line.strip().split(',')
        file = arr[0]
        labels = [name2idx[name] for name in arr[1:]]
        # print(id, labels)
        file2index[file] = labels
    return file2index
def name2index(path,encoding = 'utf-8'):
    '''
    把类别名称转换为index索引
    :param path: 文件路径
    :return: 字典
    '''
    list_name = []
    for line in open(path, encoding=encoding):
        list_name.append(line.strip())
    name2indx = {name: i for i, name in enumerate(list_name)}
    return name2indx
def split_pretrain_data(f2l):
    from sklearn.model_selection import train_test_split
    train_set, val_set = train_test_split(f2l,test_size=config.pretrain_val_radio,random_state=42)
    with open(config.pretrain_train_list,'w',encoding='utf-8') as fout1:
        for i in train_set:
            fout1.write(i)
            # fout1.write('\n')
    with open(config.pretrain_val_list,'w',encoding='utf-8') as fout2:
        for i in val_set:
            fout2.write(i)
            # fout2.write('\n')
def file2list(path):
    f2l = []
    if os.path.exists(path):
        for line in open(path,encoding='utf-8'):
            if len(line.strip()) == 0:
                continue
            f2l.append(line.strip())
    return f2l
def list2file(list,path):
    with open(path,'w',encoding='utf-8') as output:
        for i in list:
            output.write(i)
            output.write('\n')
def resample(sig, target_point_num=None):
    '''
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    '''

    sig = signal.resample(sig, target_point_num) if target_point_num else sig

    return sig

def count_labels(data, file2idx):
    '''
    统计每个类别的样本数
    :param data:
    :param file2idx:
    :return:
    '''
    cc = [0] * config.num_classes
    for fp in data:
        for i in file2idx[fp]:
            cc[i] += 1
    return np.array(cc)


def count_others(data: list, file2idx: dict) -> int:
    cnt = 0
    for rid in data:
        if len(file2idx[rid]) == 0:
            cnt += 1
    return cnt


def train(name2idx, idx2name):
    file2idx = file2index(config.train_label, name2idx)

    train, val = split_data_spv3(file2idx, file2idx2,config.val_ratio)
    wc = count_labels(train,file2idx)
    print(wc)
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx,'wc':wc}
    # print(dd)
    torch.save(dd, config.train_data)

    val_cnt = count_labels(val, file2idx)

    with open(config.train_data.split('.pth')[0] + '_cnt.csv', 'w') as output:
        output.write('idx,arraythmia,quantity,train,val\n')
        for i in range(wc.shape[0]):
            output.write(
                '{},{},{},{},{}\n'.format(i, idx2name[i], wc[i] + val_cnt[i], wc[i], val_cnt[i]))
        train_others_cnt = count_others(train, file2idx)
        val_others_cnt = count_others(val, file2idx)
        if train_others_cnt > 0 or val_others_cnt > 0:
            output.write(
                '{},{},{},{},{}\n'.format('-', '其它', train_others_cnt+val_others_cnt, train_others_cnt, val_others_cnt)
            )
        output.write(
            '全部记录,{},{},{},{}'.format(wc.shape[0], len(train) + len(val), len(train), len(val)))


def save_label(savepath, data, file2idx, idx2name):
    with open(savepath, 'w', encoding='utf-8') as output:
        for file in data:
            idxs = file2idx[file]
            output.write(file)
            for idx in idxs:
                output.write(',{}'.format(idx2name[idx]))
            output.write('\n')

def trainval_test(idx2name):
    '''
    将整个数据集划分为“训练集验证集”和“测试集两部分”，test_ratio是测试集的比例
    '''

    file2idx = file2index(config.alldata_label, name2idx)

    trainval, test = split_data_spv3(file2idx,file2idx2,config.test_ratio)
    test_cnt = count_labels(test, file2idx)
    test_others_cnt = count_others(test, file2idx)

    with open(config.test_label.split('.txt')[0] + '_cnt.csv', 'w') as output:
        output.write('idx,arraythmia,quantity\n')
        for i in range(test_cnt.shape[0]):
            output.write('{},{},{}\n'.format(i, idx2name[i], test_cnt[i]))
        if test_others_cnt > 0:
            output.write(
                '{},{},{}\n'.format('-', '其它', test_others_cnt)
            )
        output.write('总计,{},{}'.format(config.num_classes, len(test)))

    save_label(config.test_label, test, file2idx, idx2name)
    save_label(config.train_label, trainval, file2idx, idx2name)
def generate_mean_mask_index(batch,num_patches,mask_radio,device):
    num_masked = int(mask_radio*num_patches)
    num_unmasked = num_patches - num_masked
    unmasked_indices_1 = []
    masked_indices_1 = []
    step = mask_radio * 10
    # 把 0~num_patches个索引按比例分到两边
    for i in range(num_patches):
        if i % 10 >= step:
            unmasked_indices_1.append(i)
        else:
            masked_indices_1.append(i)
    unmasked_indices_1 = torch.tensor(unmasked_indices_1,dtype=torch.long,device=device)
    masked_indices_1 = torch.tensor(masked_indices_1,dtype=torch.long,device=device)
    masked_indices = masked_indices_1.repeat(batch, 1)
    unmasked_indices = unmasked_indices_1.repeat(batch, 1)
    return masked_indices,unmasked_indices

def BSW(data,band_hz = 0.5,fs=240):
    from scipy import signal
    wn1 = 2 * band_hz / fs # 只截取5hz以上的数据
    b, a = signal.butter(1, wn1, btype='high')
    filteddata = signal.filtfilt(b, a, data)
    return filteddata
def BSW(data,band_hz = 11,fs=240):#去除高频噪声
    from scipy import signal
    wn1 = 2 * band_hz / fs # 只截取11hz以上的数据
    b, a = signal.butter(1, wn1, btype='low')
    filteddata = signal.filtfilt(b, a, data)
    return filteddata
def my_output_sliding_voting(ori_output,window = 13):
    from scipy.stats import mode
    output = np.array(ori_output)
    leno = len(output)
    half_window = int(window/2)
    for index,value in enumerate(output):
        if index < half_window:
            value = mode(output[index:index+window])[0][0]
        elif index >= leno - half_window:
            value = mode(output[index-window:index])[0][0]
        else:
            value = mode(output[index-half_window:index+half_window])[0][0]
        output[index] = value
    return output
def output_sliding_voting(output,window=5):
    # window size must be odd number 奇数
    import pandas as pd
    from scipy.stats import mode
    output = pd.Series(output).rolling(window).apply(lambda x : mode(x)[0][0]).fillna(method='bfill')
    return output.values
def segment2onoff(x):
    on = []
    off = []
    diff = np.diff(x)
    on = np.argwhere(diff == 1)
    off = np.argwhere(diff == -1)
    return on,off
def onoffcount(pred_list,true_list):
    tolerate = int(240 * 0.2)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    mean_error = 0
    for pred in pred_list:
        right = 0
        for true in true_list:
            if true-tolerate <= pred <= true+tolerate:
                right = 1
                mean_error += abs(true-pred)
                TP += 1
                break
        if right == 0:
            FP +=1
    for true in true_list:
        right = 0
        for pred in pred_list:
            if true-tolerate <= pred <= true+tolerate:
                right = 1
                TN += 1
                break
        if right == 0:
            FN += 1
    return TP,TN,FP,FN,mean_error/TP
if __name__ == '__main__':
    name2idx = name2index(config.arrythmia)
    name2idx2 = name2index(config.arrythmia_ori)
    file2idx2 = file2index(config.alldata_label_ori, name2idx2)
    idx2name = {idx: name for name, idx in name2idx.items()}
    trainval_test(idx2name)
    train(name2idx, idx2name)
