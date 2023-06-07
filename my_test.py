from models.mae_1D_linearmask import MAE_linearmask
from models.ECG_mae_segmentation import ECG_mae_segmentation
from config import config
from train import val_epoch
import torch
from dataset import segment_dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from scipy.stats import mode
def output_sliding_voting(output,window=5):
    # window size must be odd number 奇数
    output = pd.Series(output).rolling(window).apply(lambda x : mode(x)[0][0]).fillna(method='bfill')
    return output.values
device = 'cpu'
pre_train_model = MAE_linearmask(pre_train=False)
state = torch.load(config.pre_train_ckpt,map_location='cpu')
pre_train_model.load_state_dict(state['state_dict'])
model = ECG_mae_segmentation(pre_train_model=pre_train_model,class_n = config.segment_class_n)
model = model.to(device)
state2 = torch.load('ckpt/mae_1D_linearmask_addUnet_202204190908_v2__lr0.001_st16to24_bsz64_datastand_True_task_segment_freeze_False/best_w.pth',map_location='cpu')
model.load_state_dict(state2['state_dict'])
seg = np.load(os.path.join(config.segment_val_dir,'ccdd8.npy'))
seg = seg.astype(np.float32)
ecg_II = (seg[0]-np.mean(seg[0]))/np.std(seg[0])
mask_arr = seg[1:,:]
x = torch.tensor(ecg_II, dtype=torch.float32)
x = x.unsqueeze(0)
y = torch.tensor(mask_arr,dtype=torch.float32)
y=y.unsqueeze(0)
print("输入形状：{}".format(x.shape))
print("标签形状：{}".format(y.shape))
print(y[0][:10])
with torch.no_grad():
    y_pred = model(x)
print("输出形状：{}".format(y_pred.shape))

print(y_pred[0,0][:10])
import torch.nn.functional as F
out_pred = F.softmax(y_pred,1).detach().cpu().numpy().argmax(axis=1)
print("输出形状2：{}".format(out_pred.shape))
print(out_pred[0][200:400])
y_pred = np.array([output_sliding_voting(i,9) for i in out_pred])
y = np.argmax(y.numpy(),axis=1)
print(y[0][200:400])
