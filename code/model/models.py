import torch
import torch.nn as nn

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

# import head
from model.head import ClsLocHead

import torch.nn as nn



class RFCLIP(nn.Module):
    def __init__(self, config):
        super(RFCLIP, self).__init__()

        self.config = config
        self.num_classes = config['model']['num_classes']
        
        #----------embedding
        if config['model']['embedding'] == 'CLIP':
            CLIPEmbed = CLIPModel(isTrainClip=False)
            if config['model']['isPreTrain'] == True :
                state_dict = torch.load(config['path']['clip_embed_path'])
                CLIPEmbed.load_state_dict(state_dict)
                # 冻结所有参数
                for param in CLIPEmbed.parameters():
                    param.requires_grad = False
            self.embedding_wifi = CLIPEmbed.embedding_wifi
            self.embedding_imu = CLIPEmbed.embedding_imu
            self.fusion = CLIPEmbed.fusion # B * 512 * L


        else: # CNN or TSSE
            self.embedding_wifi = Embedding(270)
            self.embedding_imu = Embedding(30)
            self.fusion = GatedFusionAdd2(512, isTrainClip=False)

        
        #-----------downscale
        if config['model']['embedding'] == 'TSSE':
            self.embedding = TADEmbedding_pure(512, 512, 3, 2048) 

        if config['model']['embedding'] == 'CNN' or config['model']['embedding'] == 'CLIP':
            if config['model']['downscale'] == 'cnn':
                self.Downscale = nn.ModuleList()
                for i in range(3):
                    self.Downscale.append(ds(in_channels=512)) # idea2
            elif config['model']['downscale'] == 'mamba':
                self.mambaemb = MambaEmbedding()

        #-----backbone
        if config['model']['backbone_name'] == 'TAD':
            self.backbone = TADBackbone(config)
        elif config['model']['backbone_name'] == 'mamba':
            self.backbone = Mamba(config) # to change
        else :
            assert 'backbone name error!'            
        
        #-----predict head
        layer_num = config['model']['layer_num']

        # if config['model']['backbone_name'] == 'mamba':
        #     layer_num = 5 # to change

        self.head = ClsLocHead(num_classes=30, head_layer=layer_num)
        self.priors = []
        # t = 256 # 128/256
        t = config['model']['priors']
        for i in range(layer_num):
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2

    def forward(self, X):
        wifi = X['wifi']
        imu = X['imu']
        B = imu.size(0)

        # embed & fusion
        wifi_embedd = self.embedding_wifi(wifi)
        imu_embedd = self.embedding_imu(imu)
        
        sig_embedd = self.fusion(wifi_embedd, imu_embedd)
       
        global_feat = sig_embedd.detach() # backbone

        #  # proj
        # if self.config['model']['backbone_name'] == 'mamba':
        #     sig_embedd = self.proj(sig_embedd)

        # print(global_feat.shape)

        if self.config['model']['embedding'] == 'TSSE':
            sig_embedd = self.embedding(sig_embedd)

        if self.config['model']['embedding'] == 'CNN' or self.config['model']['embedding'] == 'CLIP':
            for i in range(len(self.Downscale)):
                sig_embedd = self.Downscale[i](sig_embedd)
            # sig_embedd = self.mambaemb(sig_embedd)

        deep_feat = sig_embedd
        # print(deep_feat.shape)
       
        # backbone
        # feats = self.backbone(global_feat, deep_feat)
        if self.config['model']['backbone_name'] == 'TAD':
            feats = self.backbone(global_feat, deep_feat)
        elif self.config['model']['backbone_name'] == 'mamba':
            feats = self.backbone(deep_feat) # to change

        # predict
        out_offsets, out_cls_logits = self.head(feats)
        priors = torch.cat(self.priors, 0).to(sig_embedd.device).unsqueeze(0)
        loc = torch.cat([o.view(B, -1, 2) for o in out_offsets], 1)
        conf = torch.cat([o.view(B, -1, self.num_classes) for o in out_cls_logits], 1)

        return {
            'loc': loc,
            'conf': conf,
            'priors': priors
        }
    
class XRFMamba(nn.Module):
    def __init__(self, config) :
        super(XRFMamba, self).__init__()
        
        self.embedding_wifi = Embedding(270)
        self.embedding_imu = Embedding(30)
        self.fusion = GatedFusionWeight(512, isTrainClip=False)

        self.num_classes = config['model']['num_classes']

        self.embedding = TADEmbedding_pure(512, 512, 3, 2048) 

        self.backbone = Mamba(config)

        self.head = ClsLocHead(num_classes=30, head_layer=5)
        self.priors = []
        t = 256 
        for i in range(5):
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2
        
    def forward(self, input):
        x_imu = input['imu']
        x_wifi = input['wifi']
        B, C, L = x_imu.size()

        x_imu = self.embedding_imu(x_imu)
        x_wifi = self.embedding_wifi(x_wifi)

        x = self.fusion(x_imu, x_wifi)

        x = self.embedding(x)

        feats = self.backbone(x)

        out_offsets, out_cls_logits = self.head(feats)
        priors = torch.cat(self.priors, 0).to(x.device).unsqueeze(0)
        loc = torch.cat([o.view(B, -1, 2) for o in out_offsets], 1)
        conf = torch.cat([o.view(B, -1, self.num_classes) for o in out_cls_logits], 1)

        return {
            'loc': loc,
            'conf': conf,
            'priors': priors
        }

#--------------------------------------------------------------------------#
    

from model.XRF.mamba.blocks import ( MaskMambaBlock, MaxPooler)
class MambaEmbedding(nn.Module):
    def __init__(self):
        super(MambaEmbedding,self).__init__()
        self.emb = nn.ModuleList()
        self.branch = nn.ModuleList()
        for idx in range(2):
            self.branch.append(MaskMambaBlock(n_embd=512, n_ds_stride=2, use_mamba_type='vim'))

        
    def forward(self, x):
        B, C, L = x.size()
        mask = torch.ones(B, 1, L, dtype=torch.bool).to(x.device)
        for idx in range(len(self.branch)):
                x, mask = self.branch[idx](x, mask)
                # x2 += (x2, )
                # mask += (mask, )
        return x
