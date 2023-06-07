import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from config import config
from models.vit_1D import Transformer
import numpy as np
from data_process import generate_mean_mask_index
class Disperse_conv(nn.Module):
    def __init__(self,masking_ratio,dim1 = config.vit_patch_length,dim2 = config.vit_dim):
        super().__init__()
        self.new_dim = dim2
        self.conv = nn.Conv1d(in_channels=dim1,out_channels=dim2,kernel_size=int((1-masking_ratio)*10),stride=1,padding=int((1-masking_ratio)*10/2))

    def forward(self, tokens,masking_ratio):
        batches = tokens.shape[0]
        patch_num = tokens.shape[1]
        if not patch_num % 10 == 0:
            return tokens

        # print(unmasked_indices)
        step = int((1-masking_ratio)*10)
        new_tokens = torch.rand(batches,patch_num,self.new_dim,device=tokens.device)
        # print(new_tokens.shape)
        for i in range(1,int(patch_num/10)+1):
            conv_tokens = tokens[:,i*10-step:i*10].transpose(1,2)


            b = self.conv(conv_tokens).transpose(1,2)
            # print('-------b.shape:{},tokens.shape:{}'.format(b.shape,new_tokens[:,i*10-10:i*10].shape))

            new_tokens[:,i*10-step:i*10] = b
            
        return new_tokens

class MAE_conv(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        vit_dim,
        decoder_dim,
        masking_ratio = 0.5,
        masking_method = 'random',
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64

    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio
        self.masking_method = masking_method
        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.disperse_conv = Disperse_conv(masking_ratio=self.masking_ratio)
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        encoder_dim = vit_dim
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
        self.loss_funtion = nn.MSELoss(reduce=True,size_average=False)


    def forward(self, signal):
        device = signal.device

        # get patches
        if config.debug:
            print("-------MAE输入信号size：{}".format(signal.shape))
        patches = self.to_patch(signal)
        batch, num_patches, *_ = patches.shape

        if config.debug:
            print("-------patches size:{}".format(patches.shape))
            print("-------batch={},num_patches={}".format(batch,num_patches))
        # patch to encoder tokens and add positions



        tokens = self.patch_to_emb(patches)
        if config.debug:
            print("-------tokens without position embedding size:{}".format(tokens.shape))


        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        if config.debug:
            print("-------tokens with position embedding size:{}".format(tokens.shape))
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        # add conv

        num_masked = int(self.masking_ratio * num_patches)
        # random mask
        if self.masking_method == 'random':
            rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
            masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        elif self.masking_method == 'mean':
            masked_indices, unmasked_indices = generate_mean_mask_index(batch = batch,mask_radio=self.masking_ratio,num_patches=num_patches,device=device)
            # tokens = self.disperse_conv(tokens,unmasked_indices)
            tokens = self.disperse_conv(tokens,self.masking_ratio)
        if config.debug:
            print("-------num_mased:{}".format(num_masked))
            # print("-------rand_indeices size:{}".format(rand_indices.shape,rand_indices))

            print("-------masked_indeices size:{},unmasked_indeices size:{}".format(masked_indices.shape,unmasked_indices.shape))

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]
        if config.debug:
            print("-------unmasked tokens size:{}".format(tokens.shape))
        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)
        if config.debug:
            print("-------encoder tokens size:{}".format(encoded_tokens.shape))
        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        if config.debug:
            print("###### mask_tokens_size:{},decoderposemb_size:{}".format(mask_tokens.shape,self.decoder_pos_emb(masked_indices).shape))
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim = 1)
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[:, :num_masked]
        pred_pixel_values = self.to_pixels(mask_tokens)
        # calculate reconstruction loss


        return self.loss_funtion(pred_pixel_values, masked_patches)
