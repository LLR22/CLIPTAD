import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

def kl_loss(logits_per_image, logits_per_text, targets_image, targets_text):
    """
    logits_per_image: 图像到文本的相似度矩阵，形状为 [batch_size, batch_size]
    logits_per_text: 文本到图像的相似度矩阵，形状为 [batch_size, batch_size]
    targets_image: 图像的目标分布矩阵，形状为 [batch_size, batch_size]，每行和为1
    targets_text: 文本的目标分布矩阵，形状为 [batch_size, batch_size]，每行和为1
    """
    # 计算log softmax
    log_prob_img = F.log_softmax(logits_per_image, dim=1)
    log_prob_txt = F.log_softmax(logits_per_text, dim=1)
    
    # 计算KL散度损失，注意输入顺序是log概率在前，目标概率在后
    loss_img = F.kl_div(log_prob_img, targets_image, reduction='batchmean')
    loss_txt = F.kl_div(log_prob_txt, targets_text, reduction='batchmean')
    
    # 平均图像和文本的损失
    total_loss = (loss_img + loss_txt) / 2
    return total_loss

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


import math
def generate_target_matrix(labels, batch_size, seq_len, num_classes, device):
    multi_hot = torch.zeros((batch_size, seq_len, num_classes), device=device)
    
    for batch_idx in range(batch_size):
        for action in labels[batch_idx]:
            # 使用浮点数计算后取整
            start_f = action[0].item() * seq_len
            end_f = action[1].item() * seq_len
            
            # 新的边界计算方式
            start = int(math.floor(start_f))
            end = int(math.ceil(end_f))
            
            # 边界保护
            start = max(0, min(start, seq_len))
            end = max(0, min(end, seq_len))
            if start >= end: continue
            
            # 标记区间（左闭右开）
            multi_hot[batch_idx, start:end, int(action[2].item())] = 1

    flattened = multi_hot.view(-1, num_classes)
    return (flattened @ flattened.T > 0).float()

def normalize_target(target_matrix):
    # 添加极小值避免全零行
    target_matrix += 1e-8
    # 行归一化
    return target_matrix / target_matrix.sum(dim=1, keepdim=True)

