import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()

def get_logits( embeds1, embeds2, logit_scale):
    # 计算image_features @ text_features.T相似度矩阵
    logits_per_embeds1 = logit_scale * embeds1 @ embeds2.T
    logits_per_embeds2 = logit_scale * embeds2 @ embeds1.T
    return logits_per_embeds1, logits_per_embeds2

def cal_clip_loss(image_features, text_features, logit_scale=logit_scale):
    device = image_features.device
    logits_per_image, logits_per_text = get_logits(image_features, text_features, logit_scale)
    labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
    total_loss = (
        F.cross_entropy(logits_per_image, labels) +
        F.cross_entropy(logits_per_text, labels)
    ) / 2

    return {"contrastive_loss": total_loss}

def clip_loss2(logits_per_image, logits_per_text):

    device = logits_per_image.device

    labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
    total_loss = (
        F.cross_entropy(logits_per_image, labels) +
        F.cross_entropy(logits_per_text, labels)
    ) / 2
    
    total_loss.to(device)

    # print("total_loss: ", total_loss.device.type)
    # print(device)
    # assert total_loss.device.type == device, "total_loss"
    

    return total_loss