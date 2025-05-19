import torch
import torch.nn as nn
import torch.nn.init as init
from utils.basic_config import Config

from model.XRF.mamba.backbones import MambaBackbone
from model.TAD.backbone import TSSE, LSREF
from model.XRF.mamba.necks import FPNIdentity
from model.models import register_backbone_config, register_backbone

@register_backbone_config('mamba')
class Mamba_config(Config):
    def __init__(self, cfg = None):
        self.layer = 4
        self.n_embd = 512
        self.n_embd_ks = 3  # 卷积核大小
        self.scale_factor = 2  # 下采样率
        self.with_ln = True  # 使用 LayerNorm
        self.mamba_type = 'dbm'

        self.update(cfg)    # update ---------------------------------------------
        self.arch = (2, self.layer, 4)  # 卷积层结构：基础卷积、stem 卷积、branch 卷积

@register_backbone('mamba')
class Mamba(nn.Module):
    def __init__(self, config: Mamba_config):
        super(Mamba, self).__init__()
        # Mamba Backbone
        self.mamba_model = MambaBackbone(
            n_in=512,  # Must match the output of the embedding layer
            n_embd=config.n_embd,
            n_embd_ks=config.n_embd_ks,
            arch=config.arch,
            scale_factor=config.scale_factor,
            with_ln=config.with_ln,
            mamba_type=config.mamba_type
        )
        # Neck: FPNIdentity
        self.neck = FPNIdentity(
            in_channels=[config.n_embd] * (config.arch[-1] + 1),  # 输入特征通道，假设每一层的输出特征通道一致
            out_channel=config.n_embd,  # 输出特征通道数
            scale_factor=config.scale_factor,  # 下采样倍率
            with_ln=config.with_ln  # 是否使用 LayerNorm
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



@register_backbone_config('wifiTAD')
class WifiTAD_config(Config):
    def __init__(self, cfg = None):
        self.layer_num = 3
        self.input_length = 2048
        self.skip_ds_layer = 3
        self.priors = 128

        self.update(cfg)    # update ---------------------------------------------

@register_backbone('wifiTAD')
class WifiTAD(nn.Module):
    def __init__(self, config: WifiTAD_config):
        super(WifiTAD, self).__init__()

        self.layer_skip = config.skip_ds_layer
        self.skip_tsse = nn.ModuleList()

        self.layer_num = config.layer_num

        self.PyTSSE = nn.ModuleList()
        self.PyLSRE = nn.ModuleList()

        for i in range(self.layer_skip):
            self.skip_tsse.append(TSSE(in_channels=512, out_channels=256, kernel_size=3, stride=2,
                                       length=(config.input_length // 2) // (2 ** i)))

        for i in range(self.layer_num):
            self.PyTSSE.append(TSSE(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=config.priors//(2**i)))
            self.PyLSRE.append(LSREF(len=config.priors//(2**i),r=((config.input_length // 2)//config.priors)*(2**i)))


    def forward(self, embedd):

        deep_feat = embedd
        global_feat = embedd.detach()

        for i in range(len(self.skip_tsse)):
            deep_feat = self.skip_tsse[i](deep_feat)

        out_feats = []
        for i in range(self.layer_num):
            deep_feat = self.PyTSSE[i](deep_feat)
            out = self.PyLSRE[i](deep_feat, global_feat)
            out_feats.append(out)

        return out_feats
    