"""
基于 CLIP 预训练一个信号嵌入，看是否可以增强模型效果
"""
import torch
import torch.nn as nn
from model.VisionTransformer import VisionTransformer, ViT_wo_patch_embed, MB_ViT_v3, MB_ViT_v3_shareweight
import timm.models.vision_transformer
from functools import partial

class CNN_Embedding(nn.Module):
    """
    in: B * C * L
    out: B * 512 * L
    """
    def __init__(self, in_channels, out_channels=512, stride=1):
        super(CNN_Embedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=False),
            nn.Conv1d(128, 256, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=False),
            nn.Conv1d(256, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=False) # has debugged 
        )
        
    def forward(self, time):
        time = self.embedding(time)
        return time

class SigTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, global_pool=False, **kwargs):
        super(SigTransformer, self).__init__(**kwargs)
        self.global_pool = global_pool

        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        embed_dim=kwargs['embed_dim']
        self.pos_embed = nn.Parameter(torch.randn(1, 2048, embed_dim) * .02)##adjustable # 6-》3
    def forward_features(self, x):
        B = x.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x

from model.XRF.mamba.blocks import ( MaskMambaBlock, MaxPooler)
 
class CLIP_Embedding(nn.Module):
    def __init__(self, in_channels):
        super(CLIP_Embedding, self).__init__()
        self.embedding1 = CNN_Embedding(in_channels)
        # self.embedding2 = SigTransformer(global_pool=False, embed_dim=512, depth=1,
        #                                           num_heads=4, mlp_ratio=4, qkv_bias=True,
        #                                           norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.embedding2 = MaskMambaBlock(n_embd=512, n_ds_stride=1, use_mamba_type='vim') # 4/2
    
    def forward(self, x):
        x1 = self.embedding1(x)
        # x2 = x1.permute([0, 2, 1])
        x2 = x1 # 4/2

        B, C, L = x2.size()
        mask = torch.ones(B, 1, L, dtype=torch.bool).to(x2.device)

        x3, _ = self.embedding2(x2, mask)
        return x3 # 4/2
        return x3.permute([0, 2, 1])
        