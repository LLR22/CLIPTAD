from model.XRF.mamba.backbones import MambaBackbone
from model.TAD.backbone import TSSE, LSREF
from model.XRF.mamba.necks import FPNIdentity
import torch.nn as nn
import torch
import torch.nn.init as init

class Mamba(nn.Module):
    def __init__(self, config):
        super(Mamba, self).__init__()
        # config
        self.layer = 4
        self.n_embd = 512
        self.n_embd_ks = 3  # 卷积核大小
        self.scale_factor = 2  # 下采样率
        self.with_ln = True  # 使用 LayerNorm
        self.mamba_type = 'dbm' # dbm mychange
        self.arch = (2, self.layer, 4)  # 卷积层结构：基础卷积、stem 卷积、branch 卷积

        # Mamba Backbone
        self.mamba_model = MambaBackbone(
            n_in=512,  # Must match the output of the embedding layer
            n_embd=self.n_embd,
            n_embd_ks=self.n_embd_ks,
            arch=self.arch,
            scale_factor=self.scale_factor,
            with_ln=self.with_ln,
            mamba_type=self.mamba_type
        )

        # Neck: FPNIdentity
        self.neck = FPNIdentity(
            in_channels=[self.n_embd] * (self.arch[-1] + 1),  # 输入特征通道，假设每一层的输出特征通道一致
            out_channel=self.n_embd,  # 输出特征通道数
            scale_factor=self.scale_factor,  # 下采样倍率
            with_ln=self.with_ln  # 是否使用 LayerNorm
        )
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize neck
        for m in self.neck.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        # Initialize LayerNorm
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, L = x.size()
        batched_masks = torch.ones(B, 1, L, dtype=torch.bool).to(x.device)
        feats, masks = self.mamba_model(x, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)

        return fpn_feats

#----------------------my mamba-------------------#
from model.XRF.mamba.blocks import ( MaskMambaBlock, MaxPooler)
from model.TAD.module import ContraNorm, PoolConv, Cat_Fusion, ds
from model.TAD.backbone import  LSREF
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, length):
        super(MambaBlock, self).__init__()
        self.Downscale = ds(512)

        self.pconv = PoolConv(512)
        self.mamba = MaskMambaBlock(n_embd=512, n_ds_stride=1, use_mamba_type='vim')

        self.c1 = Cat_Fusion(1024, 1024)
        self.c2 = Cat_Fusion(1024, 1024)
        self.c3 = Cat_Fusion(2048, 512)
        self.contra_norm = ContraNorm(dim=length, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False)

    def forward(self, time, mask):
        high = self.pconv(time)
        time2 = self.Downscale(time)
        low, mamba_mask = self.mamba(time2, mask)
        high2 = self.c1(low, high)
        low2 = self.c2(high, low)
        out = self.c3(high2, low2)
        out = self.contra_norm(out)
        return out, mamba_mask


class Mamba2(nn.Module):
    def __init__(self, config):
        super(Mamba2, self).__init__()
        self.layer_num = config['model']['layer_num']
        self.priors = 128 # config['model']['priors']

        self.PyMamba = nn.ModuleList()
        self.PyLSRE = nn.ModuleList()
        # self.downsample = MaxPooler(kernel_size=3, stride=2, padding=1)
        
        for i in range(self.layer_num):
            # MaskMambaBlock(n_embd=512, n_ds_stride=1, use_mamba_type='vim')
            # self.PyMamba.append(MambaBlock(length=self.priors//(2**i)))
            self.PyMamba.append(MaskMambaBlock(n_embd=512, n_ds_stride=2, use_mamba_type='vim'))
            self.PyLSRE.append(LSREF(len=self.priors//(2**i),r=(1024//self.priors)*(2**i)))

    def forward(self, global_feat, deep_feat):
        out_feats = []

        B, C, L = deep_feat.size()
        mask = torch.ones(B, 1, L, dtype=torch.bool).to(deep_feat.device)

        for i in range(self.layer_num):
            deep_feat, mask = self.PyMamba[i](deep_feat, mask)
            mask = F.interpolate(
                            mask.to(deep_feat.dtype),          # 将mask转换为与x相同的数据类型
                            size=mask.size(-1) // 2,  # 计算目标尺寸（原始长度//步长）
                            mode='nearest'             # 使用最近邻插值
                        )
            
            out = self.PyLSRE[i](deep_feat, global_feat)
            out_feats.append(out)
        
        return out_feats