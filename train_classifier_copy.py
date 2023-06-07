from models.ECG_mae_classifier import EncoderMAE,ECG_mae_classifier
import models.resnet as resnet
import torch, time, os ,shutil
import models,utils
import numpy as np
from data_process import file2index,name2index
from torch.utils.tensorboard import SummaryWriter
from dataset import ECGDatasetForPreTrain
from torch.utils.data import DataLoader
from torch import nn,optim
from config import config
from tqdm import tqdm
import math
import pandas as pd
from scipy.stats import mode
from dataset import classifier_dataset,ECGDataset
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR,ReduceLROnPlateau
from functools import partial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda is {}".format(torch.cuda.is_available()))
print("device is {}".format(device))
torch.manual_seed(41)
torch.cuda.manual_seed(41)
def output_sliding_voting(output,window=5):
    # window size must be odd number 奇数
    output = pd.Series(output).rolling(window).apply(lambda x : mode(x)[0][0]).fillna(method='bfill')
    return output.values
def save_ckpt(state, is_best,model_save_dir):
    current_w = os.path.join(model_save_dir,config.current_w)
    best_w = os.path.join(model_save_dir,config.best_w)
    torch.save(state, current_w)
    if is_best:
        shutil.copyfile(current_w,best_w)
    epoch = state['epoch']
    if epoch % 200 == 0:
        best_10k = best_w.split('.pth')[0]+'_ep'+str(epoch)+'.pth'
        shutil.copyfile(best_w,best_10k)

def writeDataIntoExcel(xlsPath: str, data: dict):
	writer = pd.ExcelWriter(xlsPath)
	sheetNames = data.keys()  # 获取所有sheet的名称
	# sheets是要写入的excel工作簿名称列表
    
	data = pd.DataFrame(data,index=[0])
	for sheetName in sheetNames:
		data.to_excel(writer, sheet_name=sheetName)
	# 保存writer中的数据至excel
	# 如果省略该语句，则数据不会写入到上边创建的excel文件中
	writer.save()
def train(args):
    
    from timm.models.layers import trunc_normal_

    # encoder = EncoderMAE(global_pool=config.classifier_global,new_norm_layer=partial(nn.LayerNorm, eps=1e-6))
    # model = ECG_mae_classifier(pre_train_model=encoder,class_n=config.classifier_class_n)
    # if not config.pre_train_ckpt == 'None':
    #     print('train with pretrain')
    #     checkpoint = torch.load(config.pre_train_ckpt,map_location='cpu')['state_dict']
    #     msg = model.encoder.load_state_dict(checkpoint,strict=False)
    # trunc_normal_(model.fc.weight,std=0.01)
    # if config.freeze and not config.pre_train_ckpt == 'None':
    #     print('linearprobe,freeze encoder')
    #     for _,p in model.encoder.named_parameters():
    #         p.requires_grad = False

    model = resnet.resnet34_lead2()

    for p in model.named_parameters():
        print(p[0])
        print(p[1].requires_grad)
    model.to(device)
    model_save_dir = '%s/%s_%s_%s' % (
    config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"), config.experiment_name)
    if args.ex: model_save_dir += args.ex
    train_procedure(args,model,model_save_dir)
def train_procedure(args,model : nn.Module ,model_save_dir : str) -> None:
    from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR,ReduceLROnPlateau

    print(model)
    train_dataset = ECGDataset(data_path=config.train_data, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=3 if config.onserver else 1)
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=3 if config.onserver else 1)
    len_train_dataset = len(train_dataset)
    len_val_dataset = len(val_dataset)
    print("train_datasize",len_train_dataset,"val_datasize",len_val_dataset)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = config.lr)
    if config.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer,'min',patience=10,factor=0.8,min_lr=1e-8)
    if config.classifier_loss == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif config.classifier_loss == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    lr = config.lr

    min_loss = float('inf')
    start_epoch = 1
    stage =1
    summary_writer = SummaryWriter(log_dir=model_save_dir,flush_secs=2)
    shutil.copy('./config.py', os.path.join(model_save_dir, 'config.py'))
    shutil.copy('./train_classifier.py', os.path.join(model_save_dir, 'train_classifier.py'))

    for epoch in tqdm(range(start_epoch,config.max_epoch+1)):
        since = time.time()
        train_loss = train_epoch(model, optimizer, criterion,train_dataloader, show_interval=100)
        
        val_loss= val_epoch(model, criterion, val_dataloader)
        print(
            '#epoch:%02d stage:%d train_loss:%.3e val_loss:%0.3e  time:%s\n ' %
            (epoch, stage, train_loss, val_loss, utils.print_time_cost(since)))
      
        summary_writer.add_scalar('train_loss',train_loss,global_step=epoch)
        summary_writer.add_scalar('val_loss',val_loss,global_step=epoch)
       
        
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'lr': lr,
                 'stage': stage}
        save_ckpt(state,val_loss<min_loss,model_save_dir)
        min_loss = min(min_loss,val_loss)
        
        scheduler.step(val_loss)
        if config.lr_scheduler == None:
            if epoch in config.stage_epoch:
                stage +=1
                lr /=config.lr_decay
                utils.adjust_learning_rate(optimizer,lr)
                print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
        if config.debug:
            break

def train_epoch(model : nn.Module,optimizer,criterion,train_dataloader,show_interval = 10):
    model.train()
    loss_meter = 0
    it_count = 0
    for inputs,target in train_dataloader:
        inputs = inputs.to(device)
        target = target.squeeze(-1)
        optimizer.zero_grad()
        it_count +=1
        y_pred = model(inputs)
        if config.classifier_loss == 'BCE':
            loss = criterion(y_pred,target.float().to(device))
        else:
            loss = criterion(y_pred,target.to(device))
        loss.backward()
        optimizer.step()
        loss_meter += loss.cpu().item()
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e " % (it_count, loss))
        if config.debug:
            break
    return loss_meter
def val_epoch(model,criterion,val_dataloader):
    import torch.nn.functional as F
    model.eval()
    loss_meter = 0
  
    with torch.no_grad():
        for inputs,target in val_dataloader:
            inputs = inputs.to(device)
            target = target.squeeze(-1)
            y_pred = model(inputs)
            if config.classifier_loss == 'BCE':
                loss = criterion(y_pred,target.float().to(device))
            else:
                loss = criterion(y_pred,target.to(device))
            loss_meter+=loss.cpu().item()
    return loss_meter
def test(args):
    
    name2idx = name2index(config.classifier_arraythmia)
    idx2name = {idx : name for name, idx in name2idx.items()}
    true_dict = file2index(config.classifier_test_label,name2idx)
    pred_dict = dict()
    

    # pre_train_model = EncoderMAE(global_pool=config.classifier_global,new_norm_layer=partial(nn.LayerNorm, eps=1e-6))
    # pre_train_model.eval()
    # model = ECG_mae_classifier(pre_train_model=pre_train_model,class_n=config.classifier_class_n)
    model = resnet.resnet34_lead2()
    model.eval()
    # model = model.to(device)
    if args.ckpt:    
        state  = torch.load(os.path.join(args.ckpt,'best_w.pth'),map_location='cpu')
        model.load_state_dict(state['state_dict'])
    else:
        print('******未输入待测试模型参数路径********')
        return
    
    with torch.no_grad():
        test_cnt = 0
        for fid in tqdm(true_dict.keys()):
            x = np.load(os.path.join(config.classifier_data_dir,fid))
            x = x.astype(np.float32)
            if config.data_standardization:
                x = (x-np.mean(x))/np.std(x)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            x = x.unsqueeze(0)
            # x = x.expand(2,-1,-1)
            output = model(x).cpu().numpy()
            output = np.argmax(output,axis=1)
            pred_dict[fid] = output
            test_cnt+=1
            # if test_cnt>=10:
            #     break
    print('测试完毕，共测试{}条'.format(test_cnt))

    idxTP = np.zeros(len(idx2name),dtype=np.int32)
    idxTN = np.zeros(len(idx2name),dtype=np.int32)
    idxFP = np.zeros(len(idx2name),dtype=np.int32)
    idxFN = np.zeros(len(idx2name),dtype=np.int32)
    idxcnt = np.zeros(len(idx2name),dtype=np.int32)
    ana_cnt = 0
    for rid in true_dict.keys():
        true_y = true_dict[rid][0]
        pre_y = pred_dict[rid][0]
        # print('rid,true_y:{},pre_y:{}'.format(true_y,pre_y))
        for idx in range(len(idx2name)):
            # print('idx:{},label:{}'.format(idx,idx2name[idx]))
            if idx == true_y:
                idxcnt[idx] += 1
                if idx == pre_y:
                    idxTP[idx] += 1
                else:
                    idxFN[idx] += 1
            else:
                if idx == pre_y:
                    idxFP[idx] += 1
                else:
                    idxTN[idx] += 1
        ana_cnt+=1
        # print('\n')
        # if ana_cnt >=10:
        #     break

    specificity = idxTN / (idxTN + idxFP) # 预测为阴的占所有真阴性的比例
    sentitivity = idxTP / (idxTP + idxFN) #预测为阳的占所有真阳性的比例
    precision = idxTP / (idxTP + idxFP) #预测中，真阳性占所有预测阳性的比例
    accuracy = (idxTP + idxTN) / (idxTP+idxTN+idxFP+idxFN) #预测准确率
    idxF1 = 2 * precision * sentitivity / (precision+sentitivity)
    data = []
    
    # for idx in range(len(idx2name)):
    #     data_list = [idx2name[idx],specificity[idx],sentitivity[idx],precision[idx],accuracy[idx],idxF1[idx]]
    #     data.append(data_list)
    # df = pd.DataFrame(data, columns = ['label','specificity','sensitivity','precision','accuracy','idxF1'],
    #         dtype = str)
    sumTP = np.sum(idxTP)
    sumFP = np.sum(idxFP)
    sumFN = np.sum(idxFN)
    sumTN = np.sum(idxTN)
    microF1 = 2 * sumTP / (2 * sumTP + sumFP + sumFN)
    result_save_path = os.path.join(args.ckpt,'test_F1_{}.csv'.format(microF1))
    with open(result_save_path,'w',encoding='utf-8') as fout:
        fout.write('label,TrueNum,TP,TN,FP,FN,specificity,sentitivity,precision,accuracy,F1\n')
        for idx in range(len(idx2name.keys())):
            fout.write(str(idx2name[idx])+',')
            fout.write(str(idxcnt[idx])+',')
            fout.write(str(idxTP[idx])+',')
            fout.write(str(idxTN[idx])+',')
            fout.write(str(idxFP[idx])+',')
            fout.write(str(idxFN[idx])+',')
            fout.write(str(specificity[idx])+',')
            fout.write(str(sentitivity[idx])+',')
            fout.write(str(precision[idx])+',')
            fout.write(str(accuracy[idx])+',')
            fout.write(str(idxF1[idx])+'\n')
    # writeDataIntoExcel(save_path,df)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="train or infer or trans_train")
    parser.add_argument("--ckpt", type=str, help="the path of model weight file")
    parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()
    if (args.command == "train"):
        train(args)
    elif (args.command == 'test'):
        test(args)

