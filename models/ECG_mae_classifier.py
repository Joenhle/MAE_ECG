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
from torch.nn.functional import relu,softmax
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
    def __init__(self,global_pool=False, img_size=2400, patch_size=12, in_chans=1,
                 embed_dim=40, depth=12, num_heads=10,
                 mlp_ratio=2, norm_layer=nn.LayerNorm,**kwargs):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.global_pool = global_pool
        self.patch_embed = PatchEmbed_1D(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        if self.global_pool:
            new_norm_layer = kwargs['new_norm_layer']
            self.fc_norm = new_norm_layer(embed_dim)
            del self.norm
        self.initialize_weights()
        
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
    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # masking: length -> length * mask_ratio 
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) #repeat cls_token 
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:,1:,:].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        
        return outcome

class ECG_mae_classifier(nn.Module):
    def __init__(self,pre_train_model,class_n,embed_dim = 40):
        super().__init__()
        self.encoder = pre_train_model
        self.norm = torch.nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6)
        self.fc = nn.Linear(embed_dim,class_n)

    def forward(self, x):
        x = self.encoder(x)
        x = self.norm(x)
        x = self.fc(x)
        # out = relu(x)
        # out = softmax(out)
        return x
        
        
        

