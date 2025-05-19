import torch
import torch.nn as nn
import numpy as np
from model.TAD.embedding import Embedding
from model.TAD.module import ScaleExp
from model.TAD.backbone import TSSE, LSREF
from model.TAD.head import PredictionHead
from model.TAD.module import ContraNorm, PoolConv, Cat_Fusion, joint_attention, ds
from configs.config import config

class TADBackbone(nn.Module):
    def __init__(self, config):
        super(TADBackbone, self).__init__()
        self.config = config

        # self.skip_tsse = nn.ModuleList()
        # for i in range(self.layer_num):
        #     self.skip_tsse.append(TSSE(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=1024//(2**i)))

        self.PyTSSE = nn.ModuleList()
        self.PyLSRE = nn.ModuleList()

        self.priors = config['model']['priors']
        self.layer_num = config['model']['layer_num']


        for i in range(self.layer_num):
            self.PyTSSE.append(TSSE(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=self.priors//(2**i)))
            self.PyLSRE.append(LSREF(len=self.priors//(2**i),r=(1024//self.priors)*(2**i))) # 2048
        
        
        

    def forward(self, global_feat, deep_feat):
        # global_Feat: embedd
        # deep_feat: embedd->skip-tsse
        # deep_feat = embedd
        # global_feat = embedd.detach()

        # #----做一个下采样
        # for i in range(len(self.Downscale)):
        #     deep_feat = self.Downscale[i](deep_feat)
        # #-----
        # deep_feat2 = deep_feat
        # global_feat2 = deep_feat2.detach()

        # for i in range(len(self.skip_tsse)):
        #     deep_feat2 = self.skip_tsse[i](deep_feat2)

        out_feats = []

        for i in range(self.layer_num):
            deep_feat = self.PyTSSE[i](deep_feat)
            out = self.PyLSRE[i](deep_feat, global_feat)
            out_feats.append(out)
            # print("shape of each PL: ", out.shape) # my
        
        
        return out_feats