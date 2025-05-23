import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.basic_config import Config
from configs.config import config


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.sample.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max

class FocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, size_average=True):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float, int)):
            assert 0 < self.alpha < 1.0, 'alpha should be in `(0,1)`)'
            assert balance_index > -1
            alpha = torch.ones((self.num_class))
            alpha *= 1 - self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')
    def forward(self, logit, target):

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]

        # ----------memory saving way--------
        pt = logit.gather(1, target).view(-1) + self.eps  # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            self.alpha = self.alpha.to(logpt.device)

        alpha_class = self.alpha.gather(0, target.view(-1))
        logpt = alpha_class * logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def iou_loss(pred, target, weight=None, loss_type='distance-iou', reduction='none'):
    """
    Distance IoU Loss = 1 - IoU + alpha * (d / diagonal) ** 2, 
    where d is the Euclidean distance between box centers divided by diagonal.
    """
    input_offsets = pred.float()
    target_offsets = target.float()
    eps = torch.finfo(torch.float32).eps
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))
    # loss = 1.0 - iouk
    
    if weight is not None:
        loss = loss * weight.view(loss.size())
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        loss = loss.mean()
    return loss

def calc_ioa(pred, target):
    pred_left = pred[:, 0]
    pred_right = pred[:, 1]
    target_left = target[:, 0]
    target_right = target[:, 1]

    pred_area = pred_left + pred_right
    eps = torch.finfo(torch.float32).eps

    inter = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    ioa = inter / pred_area.clamp(min=eps)
    return ioa

class MultiSegmentLoss(nn.Module):
    def __init__(self, num_classes, clip_length):
        super(MultiSegmentLoss, self).__init__()
        self.num_classes = num_classes
        # self.overlap_thresh = overlap_thresh
        # self.negpos_ratio = negpos_ratio
        # self.use_gpu = use_gpu
        self.focal_loss = FocalLoss_Ori(self.num_classes, balance_index=0, size_average=False,
                                        alpha=0.1)
        self.center_loss = nn.BCEWithLogitsLoss(reduction='sum')

        self.clip_length = clip_length

    def forward(self, predictions, targets, pre_locs=None):
        """
        :param predictions: a tuple containing loc, conf and priors
        :param targets: ground truth segments and labels
        :return: loc loss and conf loss
        """
        loc_data, conf_data, priors = predictions
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
        # softmax focal loss
        conf_p = conf_data.view(-1, num_classes) 
        conf_p = conf_p.float() 
       
        targets_conf = conf_t.view(-1, 1)
        conf_p = F.softmax(conf_p, dim=1)
        # print(conf_p.shape)
        #assert False, 'test'
        # print(conf_p)
        # assert False, 'test'

        loss_c = self.focal_loss(conf_p, targets_conf)

        N = max(pos.sum(), 1)
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


