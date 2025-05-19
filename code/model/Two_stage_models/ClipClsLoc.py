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

# from model.embeddings import TextEmbedding2, classify_with_clip
from model.CLIPTAD.embedding import CLIP_Embedding
from model.XRF.mamba.blocks import ( MaskMambaBlock, MaxPooler)
# import head
from model.head import ClsLocHead, LocHead, ClsHead

import torch.nn as nn
from model.CLIPTAD.CLIPModel import CLIPModel
from model.embeddings import text_features

model_path = "/home/yanrui/code/CLIPBased_TAD/model/openai/clip-vit-base-patch32"


class ClipClsLoc(nn.Module):
    def __init__(self, config) :
        super(ClipClsLoc, self).__init__()
        self.config = config
        self.clip_length = 1500
        self.device = 'cuda'

        #--------Embedding and Fusion
        CLIPEmbed = CLIPModel(isTrainClip=False)
        state_dict = torch.load(config['path']['clip_embed_path'])
        CLIPEmbed.load_state_dict(state_dict)
        # 冻结所有参数
        for param in CLIPEmbed.parameters():
            param.requires_grad = False
        self.embedding_wifi = CLIPEmbed.embedding_wifi
        self.embedding_imu = CLIPEmbed.embedding_imu
        self.fusion = CLIPEmbed.fusion # B * 512 * L
        #--------end


        self.LocHead = LocHead(head_layer=1)
        self.ClsHead = ClsHead(num_classes=30, head_layer=1)
        self.head = ClsLocHead(num_classes=30, head_layer=1)

        self.priors = []
        t = 2048 
        layer_num = 1
        for i in range(layer_num):
            self.priors.append(
                torch.Tensor([[(c + 1) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2
        
        self.text_features = text_features.to(self.device)

    def forward(self, input):

        wifi = input['wifi']
        imu = input['imu']
                
        B, C, L = wifi.size()

        wifi = self.embedding_wifi(wifi)
        imu = self.embedding_imu(imu)
        sig = self.fusion(imu, wifi) # B,512,2048


        out_feats = []
        out_feats.append(sig)

        # out_offsets = self.LocHead(out_feats)
       
        # loc = torch.cat([o.view(B, -1, 2) for o in out_offsets], 1)

        
    
        priors = torch.cat(self.priors, 0).to(input['wifi'].device).unsqueeze(0)

        if self.config['model']['mode'] == 'test':
            # out_cls_logits = self.classify_with_clip(sig)
            # out_cls_logits = self.ClsHead(out_feats)

            out_offsets, out_cls_logits = self.head(out_feats)

            loc = torch.cat([o.view(B, -1, 2) for o in out_offsets], 1)
            conf = torch.cat([o.view(B, -1, 30) for o in out_cls_logits], 1)
           
            # assert False, 'test'
            return {
            'loc': loc,
            'conf': conf,
            'priors': priors
        }


        return  loc, priors
      
    def classify_with_clip(self, sig_emb):
        sig_emb = sig_emb.permute(0, 2, 1)
        sig_emb = sig_emb.reshape(-1, 512) # (B * 2048 )* 512
        # 加载CLIP模型
        device = 'cuda'
        # 确保输入在相同设备
        sig_emb = sig_emb.to(device)
        text_features = self.text_features.to(device)
        # 归一化图像特征（如果尚未归一化）
        sig_emb_normalized = sig_emb / sig_emb.norm(dim=1, keepdim=True)
            
        # 计算相似度矩阵（余弦相似度）
        similarity = sig_emb_normalized @ text_features.T  # [B*priors, num_classes]
            
        # 获取预测结果
        predicted_ids = torch.argmax(similarity, dim=1, keepdim=True)  # [B*priors, 1]

        # print(predicted_ids.device.type)
        return similarity
        return predicted_ids

        
