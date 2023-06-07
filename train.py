import torch, time, os ,shutil
import models,utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from dataset import ECGDatasetForPreTrain
from torch.utils.data import DataLoader
from torch import nn,optim
from config import config
from tqdm import tqdm
import math
import pandas as pd
from scipy.stats import mode
from dataset import segment_dataset,segment_dataset2
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR,ReduceLROnPlateau
from models.ECG_mae_segmentation import EncoderMAE,ECG_mae_segmentation_CNN,ECG_mae_segmentation_U_24,ECG_mae_segmentation_U_12,ECG_mae_segmentation_U_48

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
    # model
    
    from models.MultiResUnet import MultiResUnet,Unet_1D
    from models.mae_1D_linearmask import MAE_linearmask
    from models.mae_1D import MAE
    # model = MultiResUnet(in_channels=1,out_channels=config.segment_class_n).to(device)
    # model = Unet_1D(class_n = config.segment_class_n,layer_n=6)
    # encoder = EncoderMAE(patch_size=48,embed_dim=160)
    # if not config.pre_train_ckpt == 'None':
    #     state = torch.load(config.pre_train_ckpt,map_location='cpu')
    #     encoder_dict = encoder.state_dict()
    #     model_pretrained_dict = {k:v for k,v in state['state_dict'].items() if k in encoder_dict}
    #     encoder.load_state_dict(model_pretrained_dict)
    #     print("training with pre_train")
    # # model = ECG_mae_segmentation_U_12(pre_train_model=encoder,class_n=4)
    # model = ECG_mae_segmentation_U_48(pre_train_model=encoder,class_n=4)

    # if config.freeze and not config.pre_train_ckpt == 'None':
    #     print("linearprobe,freeze encoder")
    #     for _, p in model.named_parameters():
    #         p.requires_grad = True
    #     for _, p in model.encoder.named_parameters():
    #         p.requires_grad = False
    for p in model.named_parameters():
        print(p[0])
        print(p[1].requires_grad)
    model = model.to(device)
    model_save_dir = '%s/%s_%s_%s' % (
    config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"), config.experiment_name)
    if args.ex: model_save_dir += args.ex
    
    train_procedure(args,model,model_save_dir)
def train_procedure(args,model : nn.Module ,model_save_dir : str) -> None:
    from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR,ReduceLROnPlateau

    print(model)
    
    train_dataset = segment_dataset2(data_path=config.segment_train_file)
    train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=False,num_workers=6 if config.onserver else 1,drop_last=True,pin_memory=True)
    len_train_dataset = len(train_dataset)
    val_dataset = segment_dataset2(data_path=config.segment_val_file)
    val_dataloader = DataLoader(val_dataset,batch_size=config.batch_size,shuffle=False,num_workers=4 if config.onserver else 1,drop_last=True,pin_memory=True)
    len_val_dataset = len(val_dataset)
    print("train_datasize",len_train_dataset,"val_datasize",len_val_dataset)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = config.lr)
    if config.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer,'min',patience=10,factor=0.8,min_lr=1e-8)
    if config.seg_loss == 'dice_loss':
        criterion = utils.DiceLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    lr = config.lr

    min_loss = float('inf')
    start_epoch = 1
    stage =1
    summary_writer = SummaryWriter(log_dir=model_save_dir,flush_secs=2)
    shutil.copy('./config.py', os.path.join(model_save_dir, 'config.py'))
    shutil.copy('./train.py', os.path.join(model_save_dir, 'train.py'))
    for epoch in tqdm(range(start_epoch,config.max_epoch+1)):
        since = time.time()
        train_loss = train_epoch(model, optimizer, criterion,train_dataloader, show_interval=100)/len_train_dataset
        
        val_loss,all_pi,all_ri= val_epoch(model, criterion, val_dataloader)
        val_loss=val_loss/len_val_dataset
        print(
            '#epoch:%02d stage:%d train_loss:%.3e val_loss:%0.3e  time:%s\n bg_pi:%0.3f bg_ri:%0.3f p_pi:%0.3f p_ri:%0.3f r_pi:%0.3f r_ri:%0.3f t_pi:%0.3f t_ri:%0.3f' %
            (epoch, stage, train_loss, val_loss, utils.print_time_cost(since),all_pi[0],all_ri[0],all_pi[1],all_ri[1],all_pi[2],all_ri[2],all_pi[3],all_ri[3]))
        summary_writer.add_scalar('P wave F1',2*all_pi[0]*all_ri[0]/(all_pi[0]+all_ri[0]),global_step=epoch)
        summary_writer.add_scalar('R wave F1',2*all_pi[1]*all_ri[1]/(all_pi[1]+all_ri[1]),global_step=epoch)
        summary_writer.add_scalar('T wave F1',2*all_pi[2]*all_ri[2]/(all_pi[2]+all_ri[2]),global_step=epoch)        
        summary_writer.add_scalar('train_loss',train_loss,global_step=epoch)
        summary_writer.add_scalar('val_loss',val_loss,global_step=epoch)
        summary_writer.add_scalar('R wave recall',all_ri[1],global_step=epoch)
        summary_writer.add_scalar('R wave precision',all_pi[1],global_step=epoch)
        summary_writer.add_scalar('T wave recall',all_ri[2],global_step=epoch)
        summary_writer.add_scalar('T wave precision',all_pi[2],global_step=epoch)
        summary_writer.add_scalar('P wave recall',all_ri[0],global_step=epoch)
        summary_writer.add_scalar('P wave precision',all_pi[0],global_step=epoch)
        
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
        
        optimizer.zero_grad()
        it_count +=1
        y_pred = model(inputs)
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
    p_dict = [0] * 4
    r_dict = [0] * 4    
    p_sum_dict = [0] * 4
    r_sum_dict = [0] * 4
    with torch.no_grad():
        for inputs,target in val_dataloader:
            inputs = inputs.to(device)
            y_pred = model(inputs)
            loss = criterion(y_pred,target.to(device))
            loss_meter+=loss.cpu().item()
            out_pred = F.softmax(y_pred,1).detach().cpu().numpy().argmax(axis=1)
            y_pred = np.array([output_sliding_voting(i,9) for i in out_pred])
            y = np.argmax(target.numpy(),axis=1)
            for i in range(len(r_dict)):
                r_dict[i] += np.sum((y == y_pred) & (y == i))
                p_dict[i] += np.sum((y == y_pred) & (y_pred == i))
                r_sum_dict[i] += np.sum(y == i)
                p_sum_dict[i] += np.sum(y_pred == i)
    keys = ['p', 'r', 't', 'bg']
    all_pi = []
    all_ri = []
    for i in range(len(r_dict)):
        pi = np.sum(p_dict[i]) / np.sum(p_sum_dict[i])
        # print("%s-p" % keys[i], np.sum(p_dict[i]), '/', np.sum(p_sum_dict[i]), "=", pi)
        ri = np.sum(r_dict[i]) / np.sum(r_sum_dict[i])
        # print("*%s-r" % keys[i], np.sum(r_dict[i]), '/', np.sum(r_sum_dict[i]), "=", ri)
        all_pi.append(pi)
        all_ri.append(ri)
    return loss_meter,all_pi,all_ri
def test(args):
    from models.MultiResUnet import MultiResUnet,Unet_1D
    from models.mae_1D_linearmask import MAE_linearmask
    from models.ECG_mae_segmentation import ECG_mae_segmentation_CNN,ECG_mae_segmentation_U,ECG_mae_segmentation_U_12
    from models.mae_1D import MAE
    val_dataset = segment_dataset2(data_path=config.segment_val_file)
    val_dataloader = DataLoader(val_dataset,batch_size=config.batch_size,shuffle=False,num_workers=4 if config.onserver else 1,drop_last=True,pin_memory=True)
    len_val_dataset = len(val_dataset)
    print("val_datasize",len_val_dataset)
    # model = MultiResUnet(in_channels=1,out_channels=config.segment_class_n).to(device)
    # model = Unet_1D(class_n = config.segment_class_n,layer_n=6)

    # pre_train_model = MAE_linearmask(pre_train='no_decoder')
    # pre_train_model = MAE(pre_train = 'no_decoder')
    pre_train_model = EncoderMAE(patch_size=48,embed_dim=160)
    pre_train_model = pre_train_model.to(device)
    pre_train_model.eval()
    # model = ECG_mae_segmentation_U(pre_train_model=pre_train_model,class_n = config.segment_class_n)
    # model = ECG_mae_segmentation_CNN(pre_train_model=pre_train_model,class_n = config.segment_class_n)
    model = ECG_mae_segmentation_U_48(pre_train_model=pre_train_model,class_n=4)
    model = model.to(device)
    if args.ckpt:    
        state  = torch.load(os.path.join(args.ckpt,'best_w.pth'),map_location='cpu')
        model.load_state_dict(state['state_dict'])
    if config.seg_loss == 'dice_loss':
        criterion = utils.DiceLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    data = dict()
    start = time.time()
    val_loss,all_pi,all_ri= val_epoch(model, criterion, val_dataloader)
    end = time.time()
    print("cost {}s".format(end-start))
    data['P recall'] = all_ri[0]
    data['R recall'] = all_ri[1]
    data['T recall'] = all_ri[2]
    data['Bg recall'] = all_ri[3]
    data['P precision'] = all_pi[0]
    data['R precision'] = all_pi[1]
    data['T precision'] = all_pi[2]
    data['Bg precision'] = all_pi[3]
    data['P F1'] = 2*all_pi[0]*all_ri[0]/(all_pi[0]+all_ri[0])
    data['R F1'] = 2*all_pi[1]*all_ri[1]/(all_pi[1]+all_ri[1])
    data['T F1'] = 2*all_pi[2]*all_ri[2]/(all_pi[2]+all_ri[2])
    data['bg F1'] = 2*all_pi[3]*all_ri[3]/(all_pi[3]+all_ri[3])
   
    save_path = os.path.join(args.ckpt,'test.xls')
    
    writeDataIntoExcel(save_path,data)
    
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

