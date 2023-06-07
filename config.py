# -*- coding: utf-8 -*-
import os
# todo experi
# pl48 m50 tianchi ccdd
# pl48 m25 tianchi ccdd
class Config:
    onserver = True
    server = 'ecnu' # ecnu local juchi
    debug = False
    task = "segment"# pre_train segment classifier
    # fine-tune segment
    freeze = True
    
    pre_train_ckpt = 'None'
    # pre_train_experiment m25
    # pre_train_ckpt = 'ckpt_pretrain/mae_vit_signal_patch12_m25_202208201506_v2_lr0.001_bsz256_datastand_True_fs240Hz/best_w.pth'
    # pre_train_experiment m25 mean
    # pre_train_ckpt = 'ckpt_pretrain/mae_vit_signal_patch12_m25_mean_202209010456_v2_lr0.001_bsz256_datastand_True_fs240Hz/best_w.pth'
    # pre_train_experiment m50
    # pre_train_ckpt = 'ckpt_pretrain/mae_vit_signal_patch12_m50_202208190501_v2_pl48_lr0.001_24_bsz256_datastand_True_random0.5_fs240Hz/best_w.pth'
    # pre_train_experiment m50 mean
    # pre_train_ckpt = 'ckpt_pretrain/mae_vit_signal_patch12_m50_mean_202208281323_v2_lr0.001_bsz256_datastand_True_fs240Hz/best_w.pth'
    # pre_train_experiment m75
    # pre_train_ckpt = 'ckpt_pretrain/mae_vit_signal_patch12_m75_202208181236_v2_pl48_lr0.001_24_bsz256_datastand_True_random0.5_fs240Hz/best_w.pth'
    # pre_train_experiment m75 mean
    # pre_train_ckpt ='ckpt_pretrain/mae_vit_signal_patch12_m75_mean_202208271023_v2_lr0.001_bsz256_datastand_True_fs240Hz/best_w.pth'
    # pre_train_experiment m75 patch 24
    # pre_train_ckpt = 'ckpt_pretrain/mae_vit_signal_patch24_m75_202209021042_v2_lr0.001_bsz256_datastand_True_fs240Hz/best_w.pth'
    # pre_train_experiment m50 patch 24
    # pre_train_ckpt = 'ckpt_pretrain/mae_vit_signal_patch24_m50_202209030600_v2_lr0.001_bsz256_datastand_True_fs240Hz/best_w.pth'
    # pre_train_experiment m25 patch 24
    # pre_train_ckpt = 'ckpt_pretrain/mae_vit_signal_patch24_m25_202209031556_v2_lr0.001_bsz256_datastand_True_fs240Hz/best_w.pth'
    # pre_trian_experiment m75 patch 48
    # pre_train_ckpt = 'ckpt_pretrain/mae_vit_signal_patch48_m75_202209040653_v2_lr0.001_bsz256_datastand_True_fs240Hz/best_w.pth'
    # pre_trian_experiment m50 patch 48
    # pre_train_ckpt = 'ckpt_pretrain/mae_vit_signal_patch48_m50_202209041518_v2_lr0.001_bsz256_datastand_True_fs240Hz/best_w.pth'
    # pre_trian_experiment m25 patch 48
    pre_train_ckpt = 'ckpt_pretrain/mae_vit_signal_patch48_m25_202209041519_v2_lr0.001_bsz256_datastand_True_fs240Hz/best_w.pth'
    # mae_1d_4
    # pre_train_ckpt = 'ckpt_pretrain/mae_1D_202205121311_v2_pl12_lr0.001_st16to24_bsz64_datastand_True_random0.3_fs240Hz_tran4/best_w.pth'
    
    #mae_1d_12_0.3
    # pre_train_ckpt = 'ckpt_pretrain/mae_1D_202205120950_v2_pl24_lr0.001_st16to24_bsz64_datastand_True_random0.3_fs240Hz_trans12/best_w.pth'
    #mae_2d data
    image_train_data_path = '/mnt/data/MAE_2D/train'
    image_val_data_path = '/mnt/data/MAE_2D/val'
    # ------------sc正异常-----------
    if onserver:
        root = '/home/ECG_AI/shuchuang/data/20210519_120000'
    # 输入原始数据的单位值，微伏
    inputUnit_uv = 2.4
    # 送入模型里的数据的单位值，微伏
    targetUnit_uv = 4.88
    train_dir = os.path.join(root, 'alldata_npy')  # 训练集文件夹
    test_dir = os.path.join(root, 'alldata_npy')  # 测试集文件夹
    train_label = os.path.join(root, 'train_label_v19.txt')  # 训练验证集的标签
    test_label = os.path.join(root, 'test_label_v19.txt')  # 测试集的标签
    alldata_label = os.path.join(root, 'labels_v19.txt')  # 整个数据集的标签，包括测试集验证集训练集，在划分测试集时使用
    arrythmia = os.path.join(root, 'arrythmia_v19.txt')  # 类别标签文件
    train_data = os.path.join(root, 'train_v19.pth')  # 划分好的训练集验证集数据
    train_labelrelationship_matrix = os.path.join(root, 'train_v19_labelrelationship_matrix.csv')
    word_embedding_path = os.path.join(root, 'arrythmia_ori_v4_embedding.txt')
    num_classes = 1
    # ------------sc正异常-----------
    #---classifier
    classifier_class_n = 1
    classifier_loss = 'BCE'
    classifier_global = True
    classifier_root = 'D://Data/天池/TC_classifier'
    classifier_dataset_version = 'v5'
    if server == 'juchi':
        classifier_root = '/home/TC_classifier'
    elif server == 'ecnu':
        classifier_root = '/home/ECG_AI/N_MAE_ECG/TC_classifier'
    classifier_arraythmia = os.path.join(classifier_root,'arrhythmia_{}.txt'.format(classifier_dataset_version))
    classifier_data_dir = os.path.join(classifier_root,'all_data')
    
    classifier_train_data = os.path.join(classifier_root,'train_data_{}.txt'.format(classifier_dataset_version))
    classifier_val_data = os.path.join(classifier_root,'val_data_{}.txt'.format(classifier_dataset_version))
    classifier_test_data = os.path.join(classifier_root,'test_data_{}.txt'.format(classifier_dataset_version))

    classifier_train_label = os.path.join(classifier_root,'train_label_{}.txt'.format(classifier_dataset_version))
    classifier_val_label = os.path.join(classifier_root,'val_label_{}.txt'.format(classifier_dataset_version))
    classifier_test_label = os.path.join(classifier_root,'test_label_{}.txt'.format(classifier_dataset_version))

    #---
    
    #--- segment
    seg_loss = 'dice_loss' # dice_loss,ce
    segment_fs = 240
    segment_class_n = 4

    dataset = 'tianchi'

    segment_train_file = 'data/seg_{}_train.txt'.format(dataset)
    segment_val_file = 'data/seg_{}_val.txt'.format(dataset)
    segment_data_dir = 'D://Data/U-net数据/240Hz/all/baselinedriftV2_10s/alldata'
    if server == 'juchi':
        segment_data_dir = '/home/segment/alldata'
    elif server == 'ecnu':
        segment_data_dir = '/home/ECG_AI/N_MAE_ECG/baselinedriftV2_10s/alldata'
    #---
    # 本地的数据根目录
    
    pretrain_root = 'data'
    sc_data_root = '/Users/songpeidong/Documents/Data/数创静态'
    if server == 'ecnu':
        sc_data_root = '/home/ECG_AI/shuchuang/data/20210519_120000'
    elif server == 'juchi':
        sc_data_root = '/home/'
    # pre-train
    data_dir = os.path.join(sc_data_root,'alldata_npy')
    alldata_list = os.path.join(pretrain_root, 'sc_data_list.txt')  # 整个数据集的标签，包括测试集验证集训练集，在划分测试集时使用
    pretrain_val_radio = 0.2
    pretrain_train_list = os.path.join(pretrain_root,'sc_train_list_mini.txt')
    pretrain_val_list = os.path.join(pretrain_root, 'sc_val_list_mini.txt')
    if not server == 'local':
        pretrain_train_list = os.path.join(pretrain_root, 'sc_train_list_v2.txt')
        pretrain_val_list = os.path.join(pretrain_root, 'sc_val_list_v2.txt')
    data_sample_rate = 500
    target_sample_rate = 240
    input_signal_len = 2400
    model_name = 'task_error'
    if task == 'pre_train':
        model_name = 'mae_vit_signal_patch48_m25'
    elif task == 'classifier':
        model_name = 'resnet34_lead8'
        # model_name = 'resnet34_lead2_{}'.format(classifier_dataset_version)
        # model_name = 'mae_vit_signal_patch12_m75_classifier_global_{}_{}'.format(classifier_global,classifier_dataset_version)
    elif task == 'segment':
        # model_name ='mae_vit_signal_patch48_random_m25_UTrans_48'
        model_name = 'Unet1D'
    # model_name = 'mae_1D_UTrans_12' #mae_1D_UTrans_12, mae_1D_linearmask_addUnet,mae_1D_linearmask，MultiresUnet,mae_1D_linearmask_UTrans
    data_standardization = True

    # 模型参数1
    # patch的个数必须满足是10的倍数
    # vit_patch_length = 12
    # vit_dim = 8
    # vit_dim_head = 4
    # vit_depth = 4
    # vit_heads = 8
    # vit_mlp_dim = 8
    # mae_decoder_dim = 8
    # mae_masking_ratio = 0.3
    # mae_masking_method = 'random' # random,mean,block
    # mae_decoder_depth = 4
    # mae_decoder_heads = 8
    # mae_decoder_dim_head = 8
    # 模型参数2
    # patch的个数必须满足是10的倍数
    vit_patch_length = 48
    vit_dim = 20
    vit_dim_head = 6
    vit_depth = 12
    vit_heads = 8
    vit_mlp_dim = 8
    mae_decoder_dim = 12
    mae_masking_ratio = 0.5
    mae_masking_method = 'random' # random,mean,block
    mae_decoder_depth = 12
    mae_decoder_heads = 8
    mae_decoder_dim_head = 6
    # 在第几个epoch进行到下一个state,调整lr为lr/=lr_decay
    # stage_epoch = [40, 80, 120, 160,200]
    # stage_epoch = [32, 64, 128]
    # stage_epoch = [24, 48, 128, 200]
    # stage_epoch = [8, 16, 32, 48, 72]
    # stage_epoch = [8, 24, 48, 72, 96]
    stage_epoch = [16, 24, 40, 64, 80, 128]
    # 训练时的batch大小
    batch_size = 256
    if not task=='pre_train':
        batch_size =64
    # 最大训练多少个epoch
    max_epoch = 1000

    # 保存模型的文件夹
    ckpt = 'ckpt_pretrain'
    if task == 'segment':
        ckpt = 'ckpt'
    elif task== 'classifier':
        ckpt = 'ckpt_c'
    # 保存提交文件的文件夹
    sub_dir = 'submit'
    # 初始的学习率
    lr = 1e-3  # 3e-4
    if task == 'classifier':
        lr = 3e-4
    # 保存模型当前epoch的权重
    current_w = 'current_w.pth'
    # 保存最佳的权重
    best_w = 'best_w.pth'
    # 学习率衰减 lr/=lr_decay
    lr_decay = 10
    lr_scheduler = 'ReduceLROnPlateau'
        
    experiment_name = 'v2_lr{}_bsz{}_datastand_{}_fs{}Hz'.format(lr, batch_size,data_standardization,target_sample_rate)
    if not task == 'pre_train':
        experiment_name = 'v2_{}lr{}_{}_bsz{}_datastand_{}_freeze_{}_{}Hz'.format(dataset,lr,lr_scheduler,batch_size,data_standardization,freeze,target_sample_rate)
    if pre_train_ckpt == 'None' and not task == 'pre_train':
        experiment_name+='_withoutpre'
    if task == 'segment':
        experiment_name+=seg_loss
    elif task =='classifier':
        experiment_name+=classifier_loss
config = Config()

