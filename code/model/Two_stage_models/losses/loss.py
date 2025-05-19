from losses.anchor_based.anchor_based_loss import iou_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.clip import clip_loss2, kl_loss, generate_target_matrix, normalize_target
from transformers import CLIPProcessor, CLIPModel
from dataset.label.action2 import id_to_attribute

model_path = "/home/yanrui/code/CLIPBased_TAD/model/openai/clip-vit-base-patch32"

# 4/15 checkpoint

class MultiSegmentLoss(nn.Module):
    def __init__(self, num_classes, clip_length):
        super(MultiSegmentLoss, self).__init__()
        self.num_classes = num_classes
        self.clip_length = clip_length
        self.device = 'cuda'

        #----for text
        self.clip_processor = CLIPProcessor.from_pretrained(model_path)
        self.clip_encoder = CLIPModel.from_pretrained(model_path).to(self.device)
         # 完全冻结CLIP参数
        for param in self.clip_encoder.parameters():
            param.requires_grad = False
        self.text_embeddings = self._precompute_text_embeddings()
        #---- end

    def _precompute_text_embeddings(self):
        """预计算所有文本嵌入"""
        embeddings = nn.Embedding(len(id_to_attribute), 512, device=self.device)
        
        with torch.no_grad():
            for idx, desc in id_to_attribute.items():
                inputs = self.clip_processor(
                    text=desc,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                text_feature = self.clip_encoder.get_text_features(**inputs).mean(dim=0)
                embeddings.weight.data[idx] = text_feature
                
        return embeddings

    def forward(self, predictions, targets, pre_locs=None):
        """
        :param predictions: a tuple containing loc, conf and priors
        :param targets: ground truth segments and labels
        :return: loc loss and conf loss
        """ 
        sig_embed_normalized, loc_data, priors, logit_scale = predictions # !

        num_batch = len(targets)
        num_priors = priors.size(0)
        num_classes = self.num_classes

        # match priors and ground truth segments
        loc_t = torch.Tensor(num_batch, num_priors, 2).to(loc_data.device)
        conf_t = torch.LongTensor(num_batch, num_priors).to(loc_data.device)

        #-------得到真实值偏移和分类标
        with torch.no_grad():
            for idx in range(num_batch):
                truths = targets[idx][:, :-1]
                labels = targets[idx][:, -1]
                """
                match gt
                """
                # 找与真实窗口最接近的中心点
                K = priors.size(0)
                N = truths.size(0)
                center = priors[:, 0].unsqueeze(1).expand(K, N)
                left = (center - truths[:, 0].unsqueeze(0).expand(K, N)) * self.clip_length
                right = (truths[:, 1].unsqueeze(0).expand(K, N) - center) * self.clip_length
                area = left + right
                maxn = self.clip_length * 2
                area[left < 0] = maxn
                area[right < 0] = maxn
                best_truth_area, best_truth_idx = area.min(1)
                # 计算理想的便宜量应该是多少，其中loc-t存储groundtruth，conf是实际选择的ancher的label
                loc_t[idx][:, 0] = (priors[:, 0] - truths[best_truth_idx, 0]) * self.clip_length
                loc_t[idx][:, 1] = (truths[best_truth_idx, 1] - priors[:, 0]) * self.clip_length
                conf = labels[best_truth_idx]
                conf[best_truth_area >= maxn] = 0
                conf_t[idx] = conf
        #---------end

        #-------------get clip 中的 文本特征和目标矩阵
        targets_conf = conf_t.view(-1, 1)
        #-------生成目标矩阵
        # 转换标签形状: (B*L, 1) -> (B*L,)
        clip_targets = targets_conf.squeeze(-1)  # 假设每个元素是整数类别标签
        # 生成目标矩阵 (B*L, B*L)
        clip_targets = (clip_targets.unsqueeze(1) == clip_targets.unsqueeze(0)).float()
        clip_targets = normalize_target(clip_targets)
        #--------end
        targets_emb =  self.text_embeddings(targets_conf.view(-1))
        targets_emb_normalized = targets_emb / (targets_emb.norm(dim=1, keepdim=True) + 1e-8)
        #-------------end

        #----------计算loc loss
        pos = conf_t > 0  # [num_batch, num_priors]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # [num_batch, num_priors, 2]
        loc_p = loc_data[pos_idx].view(-1, 2)
        loc_target = loc_t[pos_idx].view(-1, 2)
        if loc_p.numel() > 0:
            loss_l = iou_loss(loc_p.clamp(min=0), loc_target, loss_type='liou', reduction='mean')
        else:
            loss_l = loc_p.sum()
        #----------end

        #----------计算clip的相似度
        logits_per_sig = logit_scale * sig_embed_normalized @ targets_emb_normalized.T
        logits_per_text = logit_scale * targets_emb_normalized @ sig_embed_normalized.T # (B * 449) * (B * 449)
        #---------end

       
        loss_c = kl_loss(logits_per_image=logits_per_sig, logits_per_text=logits_per_text,
                             targets_image=clip_targets, targets_text=clip_targets)

        N = max(pos.sum(), 1)
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


class LocLoss(nn.Module):
    def __init__(self, num_classes, clip_length):
        super(LocLoss, self).__init__()
        self.num_classes = num_classes
        self.clip_length = clip_length
        self.device = 'cuda'
    
    def forward(self, predictions, targets, pre_locs=None):
        """
        :param predictions: a tuple containing loc, conf and priors
        :param targets: ground truth segments and labels
        :return: loc loss and conf loss
        """ 
        loc_data, priors = predictions
        # print(f"loc_data shape: {loc_data.shape}：{priors.size(0)}")
        num_batch = loc_data.size(0)
        num_priors = priors.size(0)
        num_classes = self.num_classes
        # match priors and ground truth segments
        loc_t = torch.Tensor(num_batch, num_priors, 2).to(loc_data.device)
        conf_t = torch.LongTensor(num_batch, num_priors).to(loc_data.device)

        with torch.no_grad():
            for idx in range(num_batch):
                truths = targets[idx][:, :-1]
                labels = targets[idx][:, -1]
                """
                match gt
                """
                # 找与真实窗口最接近的中心点
                K = priors.size(0)
                N = truths.size(0)
                center = priors[:, 0].unsqueeze(1).expand(K, N)
                left = (center - truths[:, 0].unsqueeze(0).expand(K, N)) * self.clip_length
                right = (truths[:, 1].unsqueeze(0).expand(K, N) - center) * self.clip_length
                area = left + right
                maxn = self.clip_length * 2
                area[left < 0] = maxn
                area[right < 0] = maxn
                best_truth_area, best_truth_idx = area.min(1)
                # 计算理想的便宜量应该是多少，其中loc-t存储groundtruth，conf是实际选择的ancher的label
                loc_t[idx][:, 0] = (priors[:, 0] - truths[best_truth_idx, 0]) * self.clip_length
                loc_t[idx][:, 1] = (truths[best_truth_idx, 1] - priors[:, 0]) * self.clip_length
                conf = labels[best_truth_idx]
                conf[best_truth_area >= maxn] = 0
                conf_t[idx] = conf

        pos = conf_t > 0  # [num_batch, num_priors]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # [num_batch, num_priors, 2]
        loc_p = loc_data[pos_idx].view(-1, 2)
        loc_target = loc_t[pos_idx].view(-1, 2)
        if loc_p.numel() > 0:
            loss_l = iou_loss(loc_p.clamp(min=0), loc_target, loss_type='liou', reduction='mean')
        else:
            loss_l = loc_p.sum()

        N = max(pos.sum(), 1)
        loss_l /= N

        return loss_l