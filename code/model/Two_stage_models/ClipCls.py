import torch
import torch.nn as nn
import numpy as np
from model.TAD.module import  ds
# import embedding
from model.CLIPTAD.CLIPModel import CLIPModel
from model.XRF.embedding import TADEmbedding_pure
from model.TAD.embedding import Embedding
# fusion
# from model.fusion import  GatedFusionWeight
from model.XRF.fusion import GatedFusionWeight, GatedFusion, GatedFusionAdd2
# import backbone
from model.TAD.tad_backbone import TADBackbone
from model.XRF.mamba_backbone import Mamba, Mamba2

from model.embeddings import TextEmbedding2, classify_with_clip
from model.CLIPTAD.embedding import CLIP_Embedding
from model.XRF.mamba.blocks import ( MaskMambaBlock, MaxPooler)
# import head
from model.head import ClsLocHead, LocHead, ClsHead

import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel
from model.XRF.mamba.backbones import MambaBackbone
from dataset.label.action2 import id_to_attribute
from model.embeddings import text_features

model_path = "/home/yanrui/code/CLIPBased_TAD/model/openai/clip-vit-base-patch32"


class ClipCls(nn.Module):
    def __init__(self, config) :
        super(ClipCls, self).__init__()
        self.config = config
        self.clip_length = 1500
        self.device = 'cuda'

        self.embedding_wifi = CLIP_Embedding(in_channels=270) # cnn + mamba
        self.embedding_imu = CLIP_Embedding(in_channels=30) # cnn + mamba
        self.fusion = GatedFusionAdd2(512, False)

        self.Downscale = nn.ModuleList()
        for i in range(3):
            self.Downscale.append(ds(in_channels=512)) # 2048->256

        # self.backbone = Mamba(config) 
        # self.backbone = MaskMambaBlock(n_embd=512, n_ds_stride=1, use_mamba_type='dbm')
        self.backbone = TADBackbone(config)

        self.LocHead = LocHead(head_layer=3)
        # self.ClsHead = ClsHead(num_classes=30, head_layer=1)

        self.priors = []
        t = 128 # 128/256
        layer_num = 3
        for i in range(layer_num):
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.text_features = text_features
    
    import torch



    def forward(self, input):
        device = 'cuda'

        wifi = input['wifi'].to(device)
        imu = input['imu'].to(device)

        B = wifi.size(0)

        wifi = self.embedding_wifi(wifi)
        imu = self.embedding_imu(imu)
        sig = self.fusion(imu, wifi)
        global_feat = sig

        for i in range(len(self.Downscale)):
            sig = self.Downscale[i](sig)  # 2048->256

        #feats = self.backbone(sig)
        
        B, C, L = wifi.size()
        # mask = torch.ones(B, 1, L, dtype=torch.bool).to(wifi.device)
        
        # feats, _ = self.backbone(sig, mask)
        
        feats = self.backbone(global_feat, sig) # shape of feates: layer_num: [B, 256/128/64/32/16, 512]
        
        

        out_offsets = self.LocHead(feats)
        
        loc = torch.cat([o.view(B, -1, 2) for o in out_offsets], 1)
        # loc = torch.ones([B, L, 2])
    
        
        #------------CLIP------------#
        sig_embed = torch.cat(feats, dim=2) 
        sig_embed = sig_embed.reshape(-1, 512) # [B * 496, 512]
        # 归一化
        sig_embed = sig_embed / (sig_embed.norm(dim=1, keepdim=True) + 1e-8)
        #-----------end------------#

        priors = torch.cat(self.priors, 0).to(input['wifi'].device).unsqueeze(0)

        if self.config['model']['mode'] == 'test':
            
            out_cls_logits = self.classify_with_clip(sig_embed)
           
            # out_cls_logits = self.ClsHead(feats)
            # # print(conf.shape)
            conf = torch.cat([o.view(B, -1, 30) for o in out_cls_logits], 1)
           
            # assert False, 'test'
            return {
            'loc': loc,
            'conf': conf,
            'priors': priors
        }


        return sig_embed, loc, priors, self.logit_scale
      

    def classify_with_clip(self, sig_emb):
        # sig_emb = sig_emb.permute(0, 2, 1)
        # sig_emb = sig_emb.reshape(-1, 512) # (B * 2048 )* 512
        # 加载CLIP模型
        device = 'cuda'
        # 确保输入在相同设备
        sig_emb = sig_emb.to(device)
        text_features = self.text_features.to(device)
        # 归一化图像特征（如果尚未归一化）
        sig_emb_normalized = sig_emb / sig_emb.norm(dim=1, keepdim=True)
            
        # 计算相似度矩阵（余弦相似度）
        similarity = self.logit_scale * sig_emb_normalized @ text_features.T  # [B*priors, num_classes]
            
        # 获取预测结果
        predicted_ids = torch.argmax(similarity, dim=1, keepdim=True)  # [B*priors, 1]

        # print(predicted_ids.device.type)
        return similarity
        return predicted_ids

      