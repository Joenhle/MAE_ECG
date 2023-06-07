def resample(sig, target_point_num=None):
    '''
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    '''
    from scipy import signal
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig
def output_sliding_voting(output,window=5):
    # window size must be odd number 奇数
    import pandas as pd
    from scipy.stats import mode
    output=output[0]
    output = pd.Series(output).rolling(window).apply(lambda x : mode(x)[0][0]).fillna(method='bfill')
    return output.values
def plot_leadIIlist_ecg(data_dir, filelist, output_dir, rawlabelstr_dict = None,fs=500, new_fs = 240,inputUnit_uv=2.4,
                        Ridx_dict=None, Ridx_label=None, del_drift=False, band_Hz=0,model = None):
    import numpy as np
    import os
    from scipy import signal
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from pylab import mpl
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    max_list = []
    min_list = []
    datalist = []
    Nlist = []
    plist = []
    tlist = []
    figname = ""
    for i, filename in enumerate(filelist):
        if i:
            figname += ','
        figname += filename

    if os.path.exists(os.path.join(output_dir, figname + '.jpg')):
        return

    for i, filename in enumerate(filelist):
        file_path = os.path.join(data_dir, filename)
        file_path = file_path+'.xml.npy'
        data = np.load(file_path)[1]
        data = data.astype(np.float32)
        if del_drift:
            wn1 = 2 * band_Hz / fs # 只截取band_Hz以上的数据
            b, a = signal.butter(1, wn1, btype='high')
            data = signal.filtfilt(b, a, data)
        if not new_fs == 0 and not new_fs==fs:

            data = resample(data,len(data) * new_fs //fs)
        datalist.append(data)
        x = (data - np.mean(data))/np.std(data)
        x = torch.tensor(x,dtype=torch.float32)

        x = torch.unsqueeze(x,0)
        # x = torch.unsqueeze(x,0)

        target = model(x)
        out_pred = F.softmax(target, 1).detach().cpu().numpy().argmax(axis=1)
        # out_pred = np.reshape(out_pred, 2400)
        output = output_sliding_voting(out_pred,9)
        p = (output == 0)
        N = (output == 1)
        t = (output == 2)
        r = (output == 3)
        # data = data * inputUnit_uv
        p = p * 0.1
        N = N * 0.2
        t = t * 0.1
        plist.append(p)
        Nlist.append(N)
        tlist.append(t)


        max_ = -1e8
        min_ = 1e8
        if np.max(data) < 1:
            max_ = 1
        elif np.max(data) > 2.5:
            max_ = 2.5
        else:
            max_ = np.max(data)
        if np.min(data) > -1:
            min_ = -1
        elif np.min(data) < -2.5:
            min_ = -2.5
        else:
            min_ = np.min(data)

        min_list.append(min_)
        max_list.append(max_)


    mpl.rcParams['font.sans-serif'] = ['simsun']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    fig, axlist = plt.subplots(len(filelist), 1)
    if not isinstance(axlist, np.ndarray):
        axlist = [axlist]
    fig.set_size_inches((20, 4 * len(filelist)))

    for i, ax1 in enumerate(axlist):
        if not rawlabelstr_dict == None:
            if filelist[i] not in rawlabelstr_dict.keys():
                continue
        curecgy = datalist[i]
        curecgN = Nlist[i]
        curecgt = tlist[i]
        curecgp = plist[i]
        curx = np.linspace(1 / new_fs , 10, curecgy.shape[0])

        ecgline = ax1.plot(curx,curecgy,linewidth = 2.5,label = 'ECG',color = 'blue')
        ecgN = ax1.plot(curx, curecgN, linewidth = 2, label = 'QRS',color = 'red')
        ecgP = ax1.plot(curx, curecgp, linewidth = 1, label = 'P',color = 'green')
        ecgT = ax1.plot(curx, curecgt, linewidth = 1, label = 'T',color = 'cyan')

        ax1.set_ylim((min_list[i], max_list[i]))
        if rawlabelstr_dict is not None:
            ax1.set_title('{},{}'.format(filelist[i], rawlabelstr_dict[filelist[i]]), fontsize=20)
        else:
            ax1.set_title('{}'.format(filelist[i], fontsize=20))
        if Ridx_dict is not None:
            Ridxs = np.array(Ridx_dict[filelist[i]], dtype=np.int32)
            Ridxs_s = (Ridxs+1) / new_fs
            ax1.plot(Ridxs_s, curecgy[Ridxs], 'ro', label='R peaks {}'.format(Ridx_label), alpha=1)
            ax1.legend()

        xmajorLocator = MultipleLocator(0.2)
        xmajorFormatter = FormatStrFormatter('%1.2f')  # 设置x轴标签文本的格式
        xminorLocator = MultipleLocator(0.04)  # 将x轴次刻度标签设置为5的倍数

        ymajorLocator = MultipleLocator(0.5)  # 将y轴主刻度标签设置为0.5的倍数
        ymajorFormatter = FormatStrFormatter('%1.1f')  # 设置y轴标签文本的格式
        yminorLocator = MultipleLocator(0.1)  # 将此y轴次刻度标签设置为0.1的倍数

        # ax1.set_xticks(np.linspace(0, curecgy.shape[0], 51))

        ax1.xaxis.set_minor_locator(xminorLocator)
        ax1.yaxis.set_minor_locator(yminorLocator)

        # 设置主刻度标签的位置,标签文本的格式
        ax1.xaxis.set_major_locator(xmajorLocator)
        ax1.xaxis.set_major_formatter(xmajorFormatter)

        ax1.yaxis.set_major_locator(ymajorLocator)
        ax1.yaxis.set_major_formatter(ymajorFormatter)

        # 显示次刻度标签的位置,没有标签文本
        ax1.xaxis.set_minor_locator(xminorLocator)
        ax1.yaxis.set_minor_locator(yminorLocator)

        ax1.grid(axis='both', which='major', linewidth=1.3)
        ax1.grid(axis='both', which='minor', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, figname + '.jpg'))
    # plt.show()
    plt.close()

def load_labelfile(file_path: str, encoding='UTF-8') -> dict:
    label_dict = {}
    for line in open(file_path, 'r', encoding=encoding):
        if len(line.strip()) == 0:
            continue
        arr = line.strip().split(',')
        fname = arr[0]
        labels = []
        if len(arr) > 1 and arr[1] != '':
            labels = arr[1:]
        label_dict[fname] = labels
    return label_dict

def load_conclusion_file(file_path: str, encoding='GB2312') -> dict:
    rawlabelstr_dict = {}
    for line in open(file_path, 'r', encoding=encoding):
        if len(line.strip()) == 0:
            continue
        fname, rawlabelstr = line.strip().split(',')
        rawlabelstr_dict[fname] = rawlabelstr
    return rawlabelstr_dict

def load_Ridx_dict(Ridx_path: str) -> {str: list}:
    Ridx_dict = {}
    with open(Ridx_path, 'r') as cin:
        for line in cin.readlines():
            line = line.strip()
            if line:
                rid, Ridxs = line.split(',')
                Ridxs = list(map(int, Ridxs.split(' '))) if len(Ridxs) else []
                Ridx_dict[rid] = Ridxs
    return Ridx_dict


if  __name__ == '__main__':
    from tqdm import tqdm
    import os
    import random
    # data_dir = 'E:/Work/ECG_AI/上海数创医疗/data/20210519_120000/alldata_npy'
    # # typelist = ['窦性心律', '心房扑动', '心房颤动', '一度房室阻滞', '预激综合症（心室预激）', '阵发性室上性心动过速']
    # # typelist = ['心房扑动', '心房颤动', '一度房室阻滞', '预激综合症（心室预激）', '阵发性室上性心动过速']
    # typelist = ['预激综合症（心室预激）_无间歇性']
    # fs = 500

    # for cname in typelist:
    #     print('drawing {}'.format(cname))
    #     output_dir = './figure_leadII/{}'.format(cname)
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     con_path = 'conclusions_{}.txt'.format(cname)
    #     rawlabelstr_dict = load_conclusion_file(con_path, encoding='UTF-8')
    #     filelist = list(rawlabelstr_dict.keys())
    #     filelistlist = []
    #     n = len(filelist)
    #     i = 0
    #     while i < n:
    #         curfilelist = filelist[i: i + 5]
    #         filelistlist.append(curfilelist)
    #         i += 5
    #     for curfilelist in tqdm(filelistlist):
    #         plot_leadIIlist_ecg(data_dir, curfilelist, rawlabelstr_dict, output_dir, fs=fs, inputUnit_uv=2.4)

    from tqdm import tqdm
    root = 'D:\\Project\\ECG_neo4j\\ECG_AI\\data\\Total12W'
    data_dir = os.path.join(root,'alldata_npy')
    typelist = ['窦性心律', '心房扑动', '心房颤动', '一度房室阻滞', '预激综合症（心室预激）', '阵发性室上性心动过速']
    # typelist = ['心房扑动', '心房颤动', '一度房室阻滞', '预激综合症（心室预激）', '阵发性室上性心动过速']
    # typelist = ['心房扑动', '心房颤动', '一度房室阻滞', '预激综合症（心室预激）', '阵发性室上性心动过速']
    # typelist = ['心房扑动', '心房颤动', '一度房室阻滞', '预激综合症（心室预激）_无间歇性', '阵发性室上性心动过速']
    # typelist = ['心房扑动', '心房颤动', '一度房室阻滞']
    # typelist = ['预激综合症（心室预激）_无间歇性', '阵发性室上性心动过速']
    fs = 500
    new_fs = 240
    del_drift = False # False
    index_comple = False
    band_Hz = 0.5
    con_path = 'D://Project/ECG_neo4j/ECG_AI/data/Total12W/conclusions_v2.csv'  # 数据原始标签路径
    rawlabelstr_dict = load_conclusion_file(con_path, encoding='utf-8')

    sc_data_txt = 'P-QRS-T边界点-难样本'
    sc_data_txt = 'P-QRS-T边界点-难样本-早搏类'
    sc_data_file = os.path.join(root, sc_data_txt + '.txt')
    fin = open(sc_data_file)
    data_list = list(map(str, fin.read().splitlines()))
    fin.close()

    # sc_data_txt = '数创原始数据随机抽取100条'
    # sc_data_list = os.listdir(data_dir)
    # data_list = []
    # for i in [random.randint(0,len(sc_data_list)) for i in range(100)]:
    #     data_list.append(sc_data_list[i][:-4])

    # sc_data_txt = '芯跳24h数据随机切割460条'
    # hb_data_dir = 'D://Data//芯跳//24h//240hz'
    # data_list = os.listdir(hb_data_dir)

    output_path = 'D://Data/U-net数据/240Hz/figure/'
    output_dir = '{}{}{}'.format(output_path, 'sc-MAEUtrans预测结果',str(sc_data_txt)+str(band_Hz))#cname改成图片输出位置
    print('drawing {}'.format(output_dir))
    # output_dir = './figure_leadII/{}'.format(output_dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if index_comple:
        for i in range(len(data_list)):
            data_list[i]=data_list[i]+'.xml'

    filelist = data_list
    filelistlist = []
    n = len(filelist)
    i = 0
    import torch
    from config import config
    # from model.CMI_ECG_segmentation_CNV2 import CBR_1D,Unet_1D
    from models.mae_1D_linearmask import MAE_linearmask
    from models.ECG_mae_segmentation import ECG_mae_segmentation_U
    device = 'cpu'
    pre_train_model = MAE_linearmask(pre_train='dt_out')
    pre_train_model = pre_train_model.to(device)
    if config.freeze == True:
        for name, parameter in pre_train_model.named_parameters():
            parameter.requries_grad = False
    else:
        for name, parameter in pre_train_model.named_parameters():
            parameter.requries_grad = True
    state = torch.load(config.pre_train_ckpt, map_location='cpu')
    pre_train_model.load_state_dict(state['state_dict'])
    model = ECG_mae_segmentation_U(pre_train_model=pre_train_model, class_n=config.segment_class_n)
    model = model.to(device)
    model.load_state_dict((torch.load('ckpt/mae_1D_linearmask_UTrans_202205010822_v2__lr0.001_st16to24_bsz64_datastand_True_task_segment_freeze_True_240Hz/best_w.pth',map_location='cpu')['state_dict']))
    model.eval()
    while i < n:
        curfilelist = filelist[i: min(i+5,n-1)]
        filelistlist.append(curfilelist)
        i += 5
    for curfilelist in tqdm(filelistlist):
        plot_leadIIlist_ecg(data_dir, curfilelist,rawlabelstr_dict = None, output_dir=output_dir, fs=fs, new_fs=new_fs,inputUnit_uv=2.4,
        del_drift=del_drift, band_Hz=band_Hz,model = model)
        