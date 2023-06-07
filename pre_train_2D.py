import torch, time, os ,shutil
import models,utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from dataset import ECGDatasetForPreTrain
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn,optim
from config import config
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR,ReduceLROnPlateau
from models import models_mae
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(41)
torch.cuda.manual_seed(41)

def save_ckpt(state, is_best,model_save_dir):
    current_w = os.path.join(model_save_dir,config.current_w)
    best_w = os.path.join(model_save_dir,config.best_w)
    torch.save(state, current_w)
    if is_best:
        shutil.copyfile(current_w,best_w)
    epoch = state['epoch']
    if epoch % 40 == 0:
        best_10k = best_w.split('.pth')[0]+'_ep'+str(epoch)+'.pth'
        shutil.copyfile(best_w,best_10k)

def pre_train(args):
    # model
    from models.vit_2D import ViT
    from models.mae_2D import MAE

    v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
    )

    # model = MAE(
    #     encoder = v,
    #     masking_ratio = 0.75,   # the paper recommended 75% masked patches
    #     decoder_dim = 512,      # paper showed good results with just 512
    #     decoder_depth = 6       # anywhere from 1 to 8
    # )
    model = models_mae.mae_vit_base_patch16()
    model = model.to(device)
    if args.ckpt and not args.resume:
        state  = torch.load(args.ckpt,map_location='cpu')
        model.load_state_dict(state['state_dict'])
        print('train with pretrained')
    model_save_dir = '%s/%s_%s_%s' % (
    config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"), config.experiment_name)
    if args.ex: model_save_dir += args.ex
    pre_train_procedure(args,model,model_save_dir)
def pre_train_procedure(args,model : nn.Module ,model_save_dir : str) -> None:
    print(model)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_dataset = datasets.ImageFolder(config.image_train_data_path, transform=transform_train)
    train_dataloader = DataLoader(train_dataset,batch_size=128,shuffle=True,num_workers=6 if config.onserver else 1)
    len_train_dataset = len(train_dataset)
    val_dataset = datasets.ImageFolder(config.image_val_data_path, transform=transform_train)
    val_dataloader = DataLoader(val_dataset,batch_size=128,shuffle=True,num_workers=6 if config.onserver else 1)
    len_val_dataset = len(val_dataset)
    print("train_datasize",len_train_dataset,"val_datasize",len_val_dataset)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = config.lr)
    criterion = nn.MSELoss(reduce=False,size_average=False)
    scheduler = ReduceLROnPlateau(optimizer,'min',patience=10,factor=0.8,min_lr=1e-8)

    lr = config.lr

    min_loss = float('inf')
    start_epoch = 1
    stage =1
    summary_writer = SummaryWriter(log_dir=model_save_dir,flush_secs=2)
    shutil.copy('./config.py', os.path.join(model_save_dir, 'config.py'))
    pass
    for epoch in tqdm(range(start_epoch,config.max_epoch+1)):
        since = time.time()
        train_loss = train_epoch(model, optimizer, criterion,train_dataloader, show_interval=100)
        val_loss= val_epoch(model, criterion, val_dataloader)
        print(
            '#epoch:%02d stage:%d train_loss:%.3e val_loss:%0.3e  time:%s\n' %
            (epoch, stage, train_loss, val_loss, utils.print_time_cost(since)))
        summary_writer.add_scalar('train_loss',train_loss,global_step=epoch)
        summary_writer.add_scalar('val_loss',val_loss,global_step=epoch)
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'lr': lr,
                 'stage': stage}
        save_ckpt(state,val_loss<min_loss,model_save_dir)
        min_loss = min(min_loss,val_loss)
        scheduler.step(val_loss)
        # if epoch in config.stage_epoch:
        #     stage +=1
        #     # lr /=config.lr_decay
        #     utils.adjust_learning_rate(optimizer,lr)
        #     print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
        if config.debug:
            break

def train_epoch(model : nn.Module,optimizer,criterion,train_dataloader,show_interval = 10):
    model.train()
    loss_meter = 0
    it_count = 0
    for inputs in train_dataloader:
        inputs = inputs[0]
        inputs = inputs.to(device)
        optimizer.zero_grad()
        it_count +=1
        loss = model(inputs)[0].sum()
        loss.backward()
        optimizer.step()
        loss_meter += loss
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e " % (it_count, loss))
        if config.debug:
            break
    return loss_meter
def val_epoch(model,criterion,val_dataloader):
    model.eval()
    loss_meter = 0
    with torch.no_grad():
        for inputs in val_dataloader:
            inputs = inputs[0]
            inputs = inputs.to(device)
            loss = model(inputs)[0].sum()
            loss_meter+=loss
    return loss_meter

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="train or infer or trans_train")
    parser.add_argument("--ckpt", type=str, help="the path of model weight file")
    parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()
    if (args.command == "pre_train"):
        pre_train(args)

