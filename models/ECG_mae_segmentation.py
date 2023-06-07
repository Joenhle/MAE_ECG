from turtle import forward
import torch
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from config import config
from models.vit_1D import Transformer
import numpy as np
from data_process import generate_mean_mask_index
from timm.models.vision_transformer import Block

from util.pos_embed import get_1d_sincos_pos_embed
import os
class PatchEmbed_1D(nn.Module):
    """ 1D Signal to Patch Embedding
        patch_length may be the same long as embed_dim
    """ 
    def __init__(self, sig_length=2400, patch_length=12, in_chans=1, embed_dim=40, norm_layer=None, flatten=True):
        super().__init__()
        self.sig_length = sig_length
        self.patch_length = patch_length
        self.grid_size = sig_length//patch_length
        self.num_patches = self.grid_size
        self.flatten = flatten

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv1d(in_chans,embed_dim,kernel_size=patch_length,stride=patch_length)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        
        assert L == self.sig_length, 'signal length does not match.'
        x = self.proj(x)
        if self.flatten:
            # x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            x = x.transpose(1,2) # BCN -> BNC
        x = self.norm(x)
        return x
class EncoderMAE(nn.Module):
    def __init__(self, img_size=2400, patch_size=12, in_chans=1,
                 embed_dim=40, depth=12, num_heads=10,
                 mlp_ratio=2, norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed_1D(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.initialize_weights()
        if config.freeze == True:
            print('freeze')
            for parameter in self.parameters():
                parameter.requries_grad = False
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore
    def forward(self, x, mask_ratio,if_mask):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if if_mask:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) #repeat cls_token
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x_list = []
        for depth,blk in enumerate(self.blocks):
            x_list.append(x)
            x = blk(x)
        x = self.norm(x)
        x_list.append(x)
        return x_list
class CBR_1D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel=9,stride=1,padding=4):
        super().__init__()
        self.seq_list = [
        nn.Conv1d(in_channels,out_channels,kernel,stride,padding,bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()]
        
        self.seq = nn.Sequential(*self.seq_list)
        
    def forward(self,x):
        return self.seq(x)

class ECG_mae_segmentation_CNN(nn.Module):
    def __init__(self,pre_train_model,class_n):
        super().__init__()
        self.mae = pre_train_model
        self.upsample_1 = nn.ConvTranspose1d(config.mae_decoder_dim,class_n,kernel_size=8,stride=2,padding=3)
        self.upsample_2 = nn.ConvTranspose1d(class_n,class_n,kernel_size = 8,stride = 5,padding=3,output_padding=3)
        self.conv_out = CBR_1D(class_n,class_n)
        # self.down = downstream_model
        # self.upsample_1 = nn.Conv1d(config.mae_decoder_dim,config.mae_decoder_dim,kernel_size=8,stride=5,padding=6)
        # self.decoder1 = CBR_1D()
        # print(self.masking_radio)
    def forward(self, x):
        device = x.device
        decoder_tokens = self.mae(x)
        # return self.down(decoder_tokens)

        upsample_1 = self.upsample_1(decoder_tokens.transpose(1,2))

        upsample_2 = self.upsample_2(upsample_1)

        out = self.conv_out(upsample_2)

        return out

class ECG_mae_segmentation_U(nn.Module):
    def __init__(self,pre_train_model,class_n):
        super().__init__()
        self.mae = pre_train_model
        #240->480
        self.upsample_1_1 = nn.ConvTranspose1d(8,8,kernel_size=8,stride=2,padding=3)
        #480->960
        self.upsample_2_1 = nn.ConvTranspose1d(8,6,kernel_size = 9,stride = 3,padding=3)
        #960->2400
        self.upsample_3_1 = nn.ConvTranspose1d(6,class_n,kernel_size=8,stride=2,padding=3)
        self.upsample_1_2 = nn.ConvTranspose1d(8, 8, kernel_size=8, stride=2, padding=3)
        # 480->960
        self.upsample_2_2 = nn.ConvTranspose1d(8, 6, kernel_size=9, stride=3, padding=3)
        # 960->2400
        self.upsample_3_2 = nn.ConvTranspose1d(6, class_n, kernel_size=8, stride=2, padding=3)

        self.conv_out = CBR_1D(class_n,class_n)
        self.conv_cat1 = CBR_1D(8+8, 8)
        self.conv_cat2 = CBR_1D(6+6,6)
        self.conv_cat3 = CBR_1D(4+4,4)

    def forward(self, x):
        device = x.device

        decoder_tokens_list = self.mae(x)
        if config.debug:
            print(decoder_tokens_list[0].shape)
        path1 = self.upsample_1_1(decoder_tokens_list[2].transpose(1,2))
        out1 = self.upsample_1_2(decoder_tokens_list[3].transpose(1,2))
        out2 = self.conv_cat1(torch.cat([path1,out1],dim=1))

        out2 = self.upsample_2_2(out2)
        path2 = self.upsample_2_1(self.upsample_1_1(decoder_tokens_list[1].transpose(1,2)))
        out3 = self.conv_cat2(torch.cat([path2,out2],dim=1))

        out3 = self.upsample_3_2(out3)
        path3 = self.upsample_3_1(self.upsample_2_1(self.upsample_1_1(decoder_tokens_list[0].transpose(1,2))))

        out4 = self.conv_cat3(torch.cat([path3,out3],dim=1))
        out = self.conv_out(out4)
        return out
class ECG_mae_segmentation_U_12(nn.Module):
    def __init__(self,pre_train_model,class_n):
        super().__init__()
        self.encoder = pre_train_model
        #goal:200->2400
        
        self.upsample_1_1 = nn.ConvTranspose1d(40,20,kernel_size=9,stride=3,padding=3)
        self.upsample_2_1 = nn.ConvTranspose1d(20,10,kernel_size = 8,stride = 2,padding=3)
        self.upsample_3_1 = nn.ConvTranspose1d(10,6,kernel_size=7,stride=1,padding=3)
        self.upsample_4_1 = nn.ConvTranspose1d(6,class_n,kernel_size = 8,stride = 2,padding =3)
        
        self.upsample_1_2 = nn.ConvTranspose1d(40, 20, kernel_size=9, stride=3, padding=3)
        self.upsample_2_2 = nn.ConvTranspose1d(20, 10, kernel_size=8, stride=2, padding=3)
        self.upsample_3_2 = nn.ConvTranspose1d(10, 6, kernel_size=7, stride=1, padding=3)
        self.upsample_4_2 = nn.ConvTranspose1d(6,class_n,kernel_size = 8,stride = 2,padding =3)
        self.conv_out = CBR_1D(class_n,class_n)
        self.conv_cat1 = CBR_1D(20+20, 20)
        self.conv_cat2 = CBR_1D(10+10,10)
        self.conv_cat3 = CBR_1D(6+6,6)
        self.conv_cat4 = CBR_1D(4+4,4)
    def forward(self, x):
        device = x.device

        decoder_tokens_list = self.encoder(x,0.75,False)
        path1 = self.upsample_1_1(decoder_tokens_list[8][:,1:,].transpose(1,2))
        out1 = self.upsample_1_2(decoder_tokens_list[-1][:,1:,].transpose(1,2))
        out2 = self.conv_cat1(torch.cat([path1,out1],dim=1))
        
        out2 = self.upsample_2_2(out2)
        
        path2 = self.upsample_2_1(self.upsample_1_1(decoder_tokens_list[5][:,1:,].transpose(1,2)))
        out3 = self.conv_cat2(torch.cat([path2,out2],dim=1))
        
        out3 = self.upsample_3_2(out3)
        path3 = self.upsample_3_1(self.upsample_2_1(self.upsample_1_1(decoder_tokens_list[2][:,1:,].transpose(1,2))))
        
        out4 = self.conv_cat3(torch.cat([path3,out3],dim=1))
        out4 = self.upsample_4_2(out4)
        path4 = self.upsample_4_1(self.upsample_3_1(self.upsample_2_1(self.upsample_1_1(decoder_tokens_list[0][:,1:,].transpose(1,2)))))
        out5 = self.conv_cat4(torch.cat([path4,out4],dim=1))
        out = self.conv_out(out5)
        return out
class ECG_mae_segmentation_U_24(nn.Module):
    def __init__(self,pre_train_model,class_n):
        super().__init__()
        self.encoder = pre_train_model
        #goal:100->2400
        
        self.upsample_1_1 = nn.ConvTranspose1d(80,40,kernel_size=9,stride=3,padding=3)
        self.upsample_2_1 = nn.ConvTranspose1d(40,20,kernel_size = 8,stride = 2,padding=3)
        self.upsample_3_1 = nn.ConvTranspose1d(20,10,kernel_size=8,stride=2,padding=3)
        self.upsample_4_1 = nn.ConvTranspose1d(10,class_n,kernel_size = 8,stride = 2,padding =3)
        
        self.upsample_1_2 = nn.ConvTranspose1d(80, 40, kernel_size=9, stride=3, padding=3)
        self.upsample_2_2 = nn.ConvTranspose1d(40, 20, kernel_size=8, stride=2, padding=3)
        self.upsample_3_2 = nn.ConvTranspose1d(20, 10, kernel_size=8, stride=2, padding=3)
        self.upsample_4_2 = nn.ConvTranspose1d(10,class_n,kernel_size = 8,stride = 2,padding =3)
        self.conv_out = CBR_1D(class_n,class_n)
        self.conv_cat1 = CBR_1D(40+40, 40)
        self.conv_cat2 = CBR_1D(20+20,20)
        self.conv_cat3 = CBR_1D(10+10,10)
        self.conv_cat4 = CBR_1D(4+4,4)
    def forward(self, x):
        device = x.device

        decoder_tokens_list = self.encoder(x,0.75,False)
        path1 = self.upsample_1_1(decoder_tokens_list[8][:,1:,].transpose(1,2))
        out1 = self.upsample_1_2(decoder_tokens_list[-1][:,1:,].transpose(1,2))
        out2 = self.conv_cat1(torch.cat([path1,out1],dim=1))
        
        out2 = self.upsample_2_2(out2)
        
        path2 = self.upsample_2_1(self.upsample_1_1(decoder_tokens_list[5][:,1:,].transpose(1,2)))
        out3 = self.conv_cat2(torch.cat([path2,out2],dim=1))
        
        out3 = self.upsample_3_2(out3)
        path3 = self.upsample_3_1(self.upsample_2_1(self.upsample_1_1(decoder_tokens_list[2][:,1:,].transpose(1,2))))
        
        out4 = self.conv_cat3(torch.cat([path3,out3],dim=1))
        out4 = self.upsample_4_2(out4)
        path4 = self.upsample_4_1(self.upsample_3_1(self.upsample_2_1(self.upsample_1_1(decoder_tokens_list[0][:,1:,].transpose(1,2)))))
        out5 = self.conv_cat4(torch.cat([path4,out4],dim=1))
        out = self.conv_out(out5)
        return out
class ECG_mae_segmentation_U_48(nn.Module):
    def __init__(self,pre_train_model,class_n):
        super().__init__()
        self.encoder = pre_train_model
        #goal:50->2400
        
        self.upsample_1_1 = nn.ConvTranspose1d(160,80,kernel_size=10,stride=4,padding=3)
        self.upsample_2_1 = nn.ConvTranspose1d(80,40,kernel_size = 8,stride = 2,padding=3)
        self.upsample_3_1 = nn.ConvTranspose1d(40,10,kernel_size=9,stride=3,padding=3)
        self.upsample_4_1 = nn.ConvTranspose1d(10,class_n,kernel_size = 8,stride = 2,padding =3)
        
        self.upsample_1_2 = nn.ConvTranspose1d(160, 80, kernel_size=10, stride=4, padding=3)
        self.upsample_2_2 = nn.ConvTranspose1d(80, 40, kernel_size=8, stride=2, padding=3)
        self.upsample_3_2 = nn.ConvTranspose1d(40, 10, kernel_size=9, stride=3, padding=3)
        self.upsample_4_2 = nn.ConvTranspose1d(10,class_n,kernel_size = 8,stride = 2,padding =3)
        self.conv_out = CBR_1D(class_n,class_n)
        self.conv_cat1 = CBR_1D(80+80, 80)
        self.conv_cat2 = CBR_1D(40+40,40)
        self.conv_cat3 = CBR_1D(10+10,10)
        self.conv_cat4 = CBR_1D(4+4,4)
    def forward(self, x):
        device = x.device

        decoder_tokens_list = self.encoder(x,0.75,False)
        path1 = self.upsample_1_1(decoder_tokens_list[8][:,1:,].transpose(1,2))
        out1 = self.upsample_1_2(decoder_tokens_list[-1][:,1:,].transpose(1,2))
        out2 = self.conv_cat1(torch.cat([path1,out1],dim=1))
        
        out2 = self.upsample_2_2(out2)
        
        path2 = self.upsample_2_1(self.upsample_1_1(decoder_tokens_list[5][:,1:,].transpose(1,2)))
        out3 = self.conv_cat2(torch.cat([path2,out2],dim=1))
        
        out3 = self.upsample_3_2(out3)
        path3 = self.upsample_3_1(self.upsample_2_1(self.upsample_1_1(decoder_tokens_list[2][:,1:,].transpose(1,2))))
        
        out4 = self.conv_cat3(torch.cat([path3,out3],dim=1))
        out4 = self.upsample_4_2(out4)
        path4 = self.upsample_4_1(self.upsample_3_1(self.upsample_2_1(self.upsample_1_1(decoder_tokens_list[0][:,1:,].transpose(1,2)))))
        out5 = self.conv_cat4(torch.cat([path4,out4],dim=1))
        out = self.conv_out(out5)
        return out
