from model.XRF.fusion import GatedFusionWeight, GatedFusionAdd2
from model.embeddings import TextEmbedding2
from model.CLIPTAD.embedding import CLIP_Embedding

import torch.nn as nn
import torch
import numpy as np
from model.head import ClsLocHead, LocHead, ClsHead

class CLIPModel(nn.Module):
    def __init__(self, isTrainClip=True, device='cuda'):
        super(CLIPModel, self).__init__()
        # self.device = torch.device('cuda' if device=='cuda' else 'cpu') # 训练的时候需要注释掉
        self.device = device

        self.embedding_wifi = CLIP_Embedding(in_channels=270)
        self.embedding_imu = CLIP_Embedding(in_channels=30)
        self.fusion = GatedFusionAdd2(512, isTrainClip)
        # self.fusion = GatedFusionWeight(512, isTrainClip)

        self.embedding_text = TextEmbedding2(device=self.device)
        # 冻结所有TextEmbedding2的参数
        for param in self.embedding_text.parameters():
            param.requires_grad = False

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.logit_scale = torch.ones([])
        
        
        # self.to(self.device)
        # self.embedding_text.to(self.device)

    
    def forward(self, X, y):
        wifi = X['wifi']
        imu = X['imu']

        wifi_embeds = self.embedding_wifi(wifi)
        imu_embeds = self.embedding_imu(imu)

        sig_embds = self.fusion(wifi_embeds, imu_embeds) # B * 512 * 2048
        # sig_embds = sig_embds_tmp.permute(0, 2, 1) # B * 2048 * 512
        text_embeds = self.embedding_text(y) # B * 2048 * 512
 
        sig_embds = sig_embds.reshape(-1, 512) # (B * 2048 )* 512
        text_embeds = text_embeds.reshape(-1, 512) # (B * 2048 )* 512

        # 归一化 ？ to understand

        sig_embds = sig_embds / (sig_embds.norm(dim=1, keepdim=True) + 1e-8)
        text_embeds = text_embeds / (text_embeds.norm(dim=1, keepdim=True) + 1e-8)

        # 相似度计算
        logits_per_sig = self.logit_scale * sig_embds @ text_embeds.T
        logits_per_text = self.logit_scale * text_embeds @ sig_embds.T # (B * 2048) * (B * 2048)

     
        logits_per_sig = logits_per_sig.to(self.device)

        logits_per_text = logits_per_text.to(self.device)


        return logits_per_sig, logits_per_text

#-----------------------------------#
