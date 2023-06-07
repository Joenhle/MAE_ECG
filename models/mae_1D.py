import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from config import config
from models.vit_1D import Transformer
import numpy as np
import os
from data_process import generate_mean_mask_index
from pre_train import pre_train

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        x_list = []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            x_list.append(x)
        return x_list


class ViT1D(nn.Module):
    def __init__(self, *, signal_lenth=config.input_signal_len, patch_lenth=config.vit_patch_length, num_classes=1000, dim=config.vit_dim, depth=config.vit_depth, heads=config.vit_heads, mlp_dim=config.vit_mlp_dim, pool = 'cls', channels = 1, dim_head = config.vit_dim_head, dropout = 0., emb_dropout = 0.):
        super().__init__()

        assert signal_lenth % patch_lenth == 0, 'signal length must be divisible by the patch length.'

        num_patches = signal_lenth // patch_lenth
        patch_dim = channels * patch_lenth
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (sl pl) -> b sl pl',pl = patch_lenth),# sl-原始信号长度 pl-patch长度
            nn.Linear(patch_dim, dim)
            # nn.Conv1d(in_channels=patch_dim,out_channels=dim,kernel_size=1)
        )
        #o_o#
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class MAE(nn.Module):
    def __init__(
        self,
        *,
        decoder_dim = config.mae_decoder_dim,
        masking_ratio = config.mae_masking_ratio,
        masking_method = config.mae_masking_method,
        decoder_depth = config.mae_decoder_depth,
        decoder_heads = config.mae_decoder_heads,
        decoder_dim_head = config.mae_decoder_dim_head,
        pre_train = 'train'
    ):
        super().__init__()
        assert masking_ratio >=0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio
        self.masking_method = masking_method
        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.pre_train = pre_train
        self.encoder = ViT1D()
        encoder = self.encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2] # to_patch-切片，patch_to_emb-切片嵌入
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1] # ？
        # print(pixel_values_per_patch)
        # decoder parameters
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        # self.to_pixels = nn.Conv1d(decoder_dim,pixel_values_per_patch,kernel_size=1)
        self.loss_funtion = nn.MSELoss(reduce=None,size_average=False)
    def forward(self, signal,save_path = None,file_name = None):
        device = signal.device

        # get patches
       
        patches = self.to_patch(signal)
        batch, num_patches, *_ = patches.shape
        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
     
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        # random mask
        if self.masking_method == 'random':
            rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
            masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        elif self.masking_method == 'mean':
            masked_indices, unmasked_indices = generate_mean_mask_index(batch = batch,mask_radio=self.masking_ratio,num_patches=num_patches,device=device)

        # get the unmasked tokens to be encoded
        if not save_path == None:
            torch.save(patches,os.path.join(save_path,file_name+'_patches.pt'))
            torch.save(unmasked_indices, os.path.join(save_path,file_name+'_unmasked_indices.pt'))
            torch.save(masked_indices,os.path.join(save_path,file_name+'_masked_indices.pt'))
        batch_range = torch.arange(batch, device = device)[:, None]
        if not self.pre_train == 'no_decoder':
            tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

       

        # attend with vision transformer
        all_encoded_tokens = self.encoder.transformer(tokens)
        encoded_tokens = all_encoded_tokens[-1]
        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        if self.pre_train == 'no_decoder':
            return all_encoded_tokens
        decoder_tokens = self.enc_to_dec(encoded_tokens)
        masked_patches = patches[batch_range, masked_indices]
        # reapply decoder position embedding to unmasked tokens

        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim = 1)

        all_decoded_tokens = self.decoder(decoder_tokens)
        decoded_tokens = all_decoded_tokens[-1]

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[:, :num_masked]
        # 
        pred_pixel_values = self.to_pixels(mask_tokens)
        if not save_path == None:
            torch.save(masked_patches,os.path.join(save_path,file_name+'_masked_patches.pt'))
            torch.save(pred_pixel_values,os.path.join(save_path,file_name+'_pred_pixel.pt'))
        # calculate reconstruction loss
        if self.pre_train == 'train':
                return self.loss_funtion(pred_pixel_values, masked_patches)
        elif self.pre_train == 'd_out':
            return decoded_tokens
        elif self.pre_train == 'dt_out':
            return all_decoded_tokens