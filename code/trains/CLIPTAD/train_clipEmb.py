import sys
sys.path.append('/home/yanrui/code/CLIPBased_TAD')

import random
import os.path
from losses.clip import clip_loss2, kl_loss, generate_target_matrix, normalize_target
from model.CLIPTAD.CLIPModel import CLIPModel
import os
import torch
import heapq  # 用于管理最小堆
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.train.wwadl import detection_collate
from dataset.train.wwadl_multi import WWADLDatasetMutiAll
from configs.config import config 

GLOBAL_SEED = 42


def _to_var(data: dict, device):
    for key, value in data.items():
        data[key] = value.to(device)  # Directly move tensor to device
    return data

def worker_init_fn(worker_id):
    np.random.seed(GLOBAL_SEED + worker_id)

logger = logging.getLogger(__name__)

class BestModelSaver:
    def __init__(self, check_point_path, max_models=10):
        self.check_point_path = check_point_path
        self.max_models = max_models
        # 使用最大堆保存模型信息 [(负的metric, model_path), ...]
        self.best_models = []

    def save_model(self, model_state_dict, model_name, metric, is_save=False):
        # 构造保存路径
        model_path = os.path.join(self.check_point_path, f"{model_name}.pt")

        if is_save:
            torch.save(model_state_dict, model_path)
            return

        # 如果队列未满，直接保存模型
        if len(self.best_models) < self.max_models:
            torch.save(model_state_dict, model_path)
            # 保存负的metric以构造最大堆
            heapq.heappush(self.best_models, (-metric, model_path))
            print(f"Model saved: {model_path} (Metric: {metric:.5f})")
        else:
            # 检查是否优于当前最差模型（堆顶是负的最大值，对应正的最小值）
            if metric < -self.best_models[0][0]:  # 假设指标是损失，越小越好
                # 删除最差模型
                _, worst_model_path = heapq.heappop(self.best_models)
                if os.path.exists(worst_model_path):
                    os.remove(worst_model_path)
                    print(f"Old model removed: {worst_model_path}")

                # 保存新模型
                torch.save(model_state_dict, model_path)
                heapq.heappush(self.best_models, (-metric, model_path))
                print(f"Model saved: {model_path} (Metric: {metric:.5f})")
            else:
                print(f"Model not saved. Metric: {metric:.5f} is worse than the top 10.")

    def get_best_models(self):
        # 返回按指标从小到大排序的模型列表（还原负的metric）
        return sorted([(-metric, path) for metric, path in self.best_models], key=lambda x: x[0])

class Trainer_ForCLIP(object):
    def __init__(self,
                 train_dataset,
                 model
                 ):
        super(Trainer_ForCLIP, self).__init__()
        self.model = model
        self.train_dataset = train_dataset

        self.batch_size = 8
        self.num_epoch = 100
        self.lr_rate_adjust_epoch = 30
        self.lr_rate_adjust_factor = 0.5
        # loss setting -----------------------------------------------------------
        if config['training']['clip_emb_loss'] == 'cross_entropy':
            self.loss = clip_loss2
        elif config['training']['clip_emb_loss'] == 'kl':
            self.loss = kl_loss
        else:
            assert 'wrong loss name!'

        # learning config ---------------------------------------------------------
        self.opt_method = 'adamw'
        self.lr_rate = 4e-4
        self.weight_decay = 1e-3

        self.check_point_path = '/home/yanrui/code/CLIPBased_TAD/model/CLIPTAD/preTrain_Emb/'

        self.model_info = 'clipEmbedding_withMamba_GatefusionWeight'
        self.writer = SummaryWriter(os.path.join(self.check_point_path, f'tb_{self.model_info}'))

        # DDP setting -------------------------------------------------------------
        self.dist_url = 'env://'
        self.rank = 0
        self.world_size = 0
        self.gpu=0

        self.device = 'cuda'
    
    def _init_optimizer(self):

        params = self.model.parameters()

        if self.opt_method == 'adam':
            self.optimizer = torch.optim.Adam(params=params,
                                              lr=self.lr_rate,
                                              weight_decay=self.weight_decay)
        elif self.opt_method == 'adamw':
            self.optimizer = torch.optim.AdamW(params=params,
                                               lr=self.lr_rate,
                                               weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(params=params,
                                             lr=self.lr_rate,
                                             weight_decay=self.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         self.lr_rate_adjust_epoch,
                                                         self.lr_rate_adjust_factor)

    def _train_one_step(self, data, targets):

        data = _to_var(data, self.device) # 将本来是字典的数据移动到设备上

        # data = data.to(self.device)  # 确保输入数据在正确的设备上
        targets = [t.to(self.device) for t in targets] # 将label移动到设备上
        self.optimizer.zero_grad()


        try:
            sig_emb, text_emb = self.model(data, targets)
            tmp_sig_emb, tmp_text_emb = sig_emb, text_emb
            
        except AssertionError as e:
            logging.info("Error occurred during training, saving data and model parameters for debugging...")

            # 创建保存路径
            error_save_path = os.path.join(self.check_point_path, "debug")
            os.makedirs(error_save_path, exist_ok=True)

            # 保存导致问题的输入数据
            data_save_path = os.path.join(error_save_path, "error_data.pt")
            torch.save(data, data_save_path)
            logging.info(f"Input data saved to: {data_save_path}")

            # 保存模型参数
            model_save_path = os.path.join(error_save_path, "error_model.pth")
            torch.save(self.model.state_dict(), model_save_path)
            logging.info(f"Model parameters saved to: {model_save_path}")

            # 再次抛出异常以中断训练
            raise e

        if config['training']['clip_emb_loss'] == 'cross_entropy':
            loss = self.loss(tmp_sig_emb, tmp_text_emb)
        elif config['training']['clip_emb_loss'] == 'kl':
            clip_targets =  generate_target_matrix(labels=targets, batch_size=len(targets), 
                                                   seq_len=config['dataset']['data_length'],
                                                   num_classes=config['model']['num_classes'],
                                                   device=self.device)
            clip_targets_normalized = normalize_target(clip_targets)
            loss = self.loss(logits_per_image=tmp_sig_emb, logits_per_text=tmp_text_emb,
                             targets_image=clip_targets_normalized, targets_text=clip_targets_normalized)

        # 反向传播
        # loss.backward(retain_graph=True)
        loss.backward()

        # 优化器更新权重
        self.optimizer.step()

        # 无需分布式同步，直接返回损失
        return loss.item()
    
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic =True

    def training(self):
        print('loss type: ', config['training']['clip_emb_loss'])
        # 给不同的进程分配不同的、固定的随机数种子
        self.set_seed(2024)
        # register_hooks(self.model) # 钩子函数？不懂，后面去研究
        device = torch.device(self.device)

        # dataset loader -------------------------------------------------------------------------------
        nw = min([os.cpu_count(), self.batch_size if self.batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,  # DataParallel 不需要使用分布式采样
            pin_memory=True,
            num_workers=nw,  # 动态设置 workers
            collate_fn=detection_collate,  # 自定义 collate_fn
            worker_init_fn=worker_init_fn,  # 初始化每个 worker 的随机种子
            drop_last = True
        )

        # load model ------------------------------------------------------------------------------------

        # 转为DataParallel模型 ---------------------------------------------------------------------------
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        else:
            print("Using a single GPU for training")

        self.model = self.model.to(device=device)

        self._init_optimizer()

        mini_train_loss = float('inf')
        saver = BestModelSaver(self.check_point_path, max_models=1)  # 初始化最佳模型管理

        for epoch in range(self.num_epoch):
            np.random.seed(epoch)  # 设置随机种子
            self.model.train()

            tbar = tqdm(train_loader)

            iteration = 0
            cost_val = 0
            loss_val = 0

            for clips, targets in tbar:
                iteration += 1
                
                loss = self._train_one_step(clips, targets)

                loss_val += loss
                cost_val += loss

                tbar.set_description('Epoch: %d: ' % (epoch + 1))
                tbar.set_postfix(train_loss=loss)
                # 每次迭代清理显存
                torch.cuda.empty_cache()


            tbar.close()

            loss_val /= (iteration + 1)
            cost_val /= (iteration + 1)
            plog = 'Epoch-{} Loss: Total - {:.5f}, loc - {:.5f}' \
                .format(epoch, cost_val, loss_val)

            logging.info(plog)

            self.scheduler.step()

            # if epoch == 39:
            #     saver.save_model(self.model.state_dict(), f"{self.model_info}_39-epoch-{epoch}", cost_val, is_save=True)
            # if epoch == 64:
            #     saver.save_model(self.model.state_dict(), f"{self.model_info}_64-epoch-{epoch}", cost_val, is_save=True)
            
            if epoch == 80:
                saver.save_model(self.model.state_dict(), f"{self.model_info}_80-epoch-{epoch}", cost_val, is_save=True)


            # 保存当前模型
            saver.save_model(self.model.state_dict(), f"{self.model_info}-epoch-{epoch}", cost_val)

            self.writer.add_scalar("Train Loss", cost_val, epoch)
            self.writer.add_scalar("loss_val Loss", loss_val, epoch)
        

import os 
os.environ['CUDA_VISIBLE_DEVICES']='1' 

if __name__ == '__main__':
    #-----import data
    train_dataset_multi = WWADLDatasetMutiAll(config['path']['train_data_path'], split='train')
    #-----import model
    model = CLIPModel()
    # path_checkpoint = '/home/yanrui/code/CLIPBased_TAD/model/CLIPTAD/preTrain_Emb/clipEmbedding-epoch-4.pt'
    # checkpoint = torch.load(path_checkpoint, map_location = torch.device('cpu'))
    # model.load_state_dict(checkpoint,strict=True)
    # model.eval()  # 先设为评估模式
    # model.
    #-----train
    trainer = Trainer_ForCLIP(train_dataset_multi, model)
    trainer.training()
