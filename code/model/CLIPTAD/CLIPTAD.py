from model.CLIPTAD.CLIPModel import CLIPModel
from model.TAD.tad_backbone import TADBackbone
import torch
import torch.nn as nn


class CLIPTAD(nn.Module):
    def __init__(self, config):
        super(CLIPTAD, self).__init__()
        self.config = config
        #---------加载预训练好的wifi-imu信号编码器并冻结权重
        CLIPEmbed = CLIPModel(config)
        state_dict = torch.load(config['path']['clip_embed_path'])
        CLIPEmbed.load_state_dict(state_dict)
        # 冻结所有参数
        for param in CLIPEmbed.parameters():
            param.requires_grad = False
        self.Embed_wifi = CLIPEmbed.embedding_wifi
        self.Embed_imu = CLIPEmbed.embedding_imu
        self.Fusion = CLIPEmbed.fusion
        #--------- backbone
        if config['model']['backbone_name'] == 'TAD':
            self.Backbone = TADBackbone(config)
        else:
            assert 'error!'
        #--------- prediction head
    
    def forward(self, X):
        #----get data
        wifi = X['wifi']
        imu = X['imu']
        #----Embedding
        
        wifi_embedd = self.Embed_wifi(wifi)
        # imu_embedd = self.Embed_imu(imu)
        # wifi_embedd = wifi_embedd.permute(0, 2, 1)
        # imu_embedd = imu_embedd.permute(0, 2, 1)

        # sig_embedd = self.Fusion(wifi_embedd, imu_embedd)
        # sig_embedd = sig_embedd.permute(0, 2, 1)
        sig_embedd = wifi_embedd
        #----backbone and predict
        loc, conf, priors = self.Backbone(sig_embedd)
        return {
            'loc': loc,
            'conf': conf,
            'priors': priors
        }
