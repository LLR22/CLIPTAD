import json
import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class WWADLDatasetSingle(Dataset):
    """
    单一模态的数据集
    """
    def __init__(self, dataset_dir, split="train", normalize=True, modality = None, device_keep_list=None):
        """
        初始化数据集
        : param dataset_dir: 数据集所在目录路径。
        : param split: 数据集分割，"train" 或 "test"。
        """
        assert split in ["train", "test"], "split must be 'train' or 'test'"
        self.dataset_dir = dataset_dir
        self.split = split
        self.normalize = normalize

        self.data_path = os.path.join(dataset_dir, f"{split}_data.h5")
        self.label_path = os.path.join(dataset_dir, f"{split}_label.json")
        self.info_path = os.path.join(dataset_dir, f"info.json")
        # self.stats_path = os.path.join(dataset_dir, "global_stats.json")  # 保存均值和方差的路径
        self.stats_path = os.path.join('/home/yanrui/code/CLIPBased_TAD/dataset/cache/', "global_stats.json")  # 保存均值和方差的路径
        
        self.device_keep_list = device_keep_list

        with open(self.info_path, 'r') as json_file:
            self.info = json.load(json_file)

        if modality is None:
            assert len(self.info['modality_list']) == 1, "single modality"
            self.modality = self.info['modality_list'][0]
        else:
            self.modality = modality

        with open(self.label_path, 'r') as json_file:
            self.labels = json.load(json_file)[self.modality]

        # 加载或计算全局均值和方差
        if self.normalize:
            if os.path.exists(self.stats_path):
                self.load_global_stats()  # 文件存在则加载
            elif split == "train":
                self.global_mean, self.global_std = self.compute_global_mean_std()  # 训练集计算
                self.save_global_stats()  # 保存均值和方差
            else:
                raise FileNotFoundError(f"{self.stats_path} not found. Please generate it using the training dataset.")

    def compute_global_mean_std(self):
        """
        计算全局均值和方差，针对序列维度计算。
        """
        print("Calculating global mean and std...")
        mean_list, std_list = [], []
        with h5py.File(self.data_path, 'r') as h5_file:
            data = h5_file[self.modality]
            for i in tqdm(range(data.shape[0]), desc="Processing samples"):
                sample = data[i]

                if self.modality == 'imu':
                    # IMU 数据：转换为 [30, 2048]
                    sample = sample.transpose(1, 2, 0).reshape(-1, sample.shape[0])  # [30, 2048]

                if self.modality == 'wifi':
                    # WiFi 数据：转换为 [270, 2048]
                    sample = sample.transpose(1, 2, 3, 0).reshape(-1, sample.shape[0])  # [270, 2048]
                
                if self.modality == 'airpods':
                    acceleration = sample[:, 3:6]  # 加速度: X, Y, Z (列索引 3 到 5)
                    rotation = sample[:, 6:9]      # 角速度: X, Y, Z (列索引 6 到 8)
                    # 将加速度和角速度合并成新的数组
                    sample = np.hstack((acceleration, rotation)).reshape(-1, sample.shape[0])

                # 转为 torch.Tensor
                sample = torch.tensor(sample, dtype=torch.float32)

                # 针对序列维度（即第 0 维）计算均值和方差
                mean_list.append(sample.mean(dim=1).numpy())  # 每个样本的序列均值，形状为 [270] 或 [30]
                std_list.append(sample.std(dim=1).numpy())  # 每个样本的序列标准差，形状为 [270] 或 [30]

        # 对所有样本的均值和标准差进行平均
        global_mean = np.mean(mean_list, axis=0)  # 最终均值，形状为 [270] 或 [30]
        global_std = np.mean(std_list, axis=0)  # 最终标准差，形状为 [270] 或 [30]

        return global_mean, global_std

    def save_global_stats(self):
        """
        保存全局均值和方差到文件。
        """
        stats = {
            self.modality: {
                "global_mean": self.global_mean.tolist(),
                "global_std": self.global_std.tolist()
            }
        }
        with open(self.stats_path, 'w') as f:
            json.dump(stats, f)
        print(f"Global stats saved to {self.stats_path}")

    def load_global_stats(self):
        """
        从文件加载全局均值和方差。
        如果文件中不存在当前 modality，则计算并更新文件。
        """
        with open(self.stats_path, 'r') as f:
            stats = json.load(f)

        # 如果当前 modality 不在文件中，计算并保存
        if self.modality not in stats:
            print(f"Modality '{self.modality}' not found in stats file. Computing and updating...")
            global_mean, global_std = self.compute_global_mean_std()
            stats[self.modality] = {
                "global_mean": global_mean.tolist(),
                "global_std": global_std.tolist()
            }
            with open(self.stats_path, 'w') as f:
                json.dump(stats, f)
            print(f"Updated global stats saved to {self.stats_path}")

        # 从文件中加载当前 modality 的均值和方差
        self.global_mean = np.array(stats[self.modality]["global_mean"])
        self.global_std = np.array(stats[self.modality]["global_std"])

    def shape(self):
        with h5py.File(self.data_path, 'r') as h5_file:
            data = h5_file[self.modality]
            shape = data.shape
        return shape

    def __len__(self):
        """
        返回数据集的样本数。
        """
        with h5py.File(self.data_path, 'r') as h5_file:
            data_length = h5_file[self.modality].shape[0]
        return data_length

    def __getitem__(self, idx):
        # 获取数据和标签
        # 延迟加载 h5 文件
        with h5py.File(self.data_path, 'r') as h5_file:
            sample = h5_file[self.modality][idx]

        label = self.labels[str(idx)]

        # 转换为 Tensor
        sample = torch.tensor(sample, dtype=torch.float32)

        # 根据模态处理数据
        if self.modality == 'imu':
            sample = self.process_imu(sample)
        elif self.modality == 'wifi':
            sample = self.process_wifi(sample)
        elif self.modality == 'airpods':
            sample = self.process_airpods(sample)

        if torch.isnan(sample).any() or torch.isinf(sample).any():
            raise ValueError("Input contains NaN or Inf values.")

        label = torch.tensor(label, dtype=torch.float32)

        # 将类别部分（最后一列）单独转换为整数
        label[:, -1] = label[:, -1].to(torch.long)  # 或者直接使用 label[:, -1] = label[:, -1].int()

        data = {
            self.modality: sample
        }

        return data, label
    
    def process_imu(self, sample):
        sample = sample.permute(1, 2, 0)  # [5, 6, 2048]
        device_num = sample.shape[0]
        imu_channel = sample.shape[1]
        sample = sample.reshape(-1, sample.shape[-1])  # [5*6=30, 2048]
        # 全局归一化：使用序列维度的均值和标准差
        if self.normalize:
            sample = (sample - torch.tensor(self.global_mean, dtype=torch.float32)[:, None]) / \
                     (torch.tensor(self.global_std, dtype=torch.float32)[:, None] + 1e-6)
            
        if self.device_keep_list:
            # 1. sample [5*6=30, 2048] -> [5, 6, 2048]
            sample = sample.reshape(device_num, imu_channel, -1)  # [5, 6, 2048]
            # 2. 保留设备 [5, 6, 2048] -> [2, 6, 2048] 保留设备 2, 3
            sample = sample[self.device_keep_list]  # [len(device_keep_list), 6, 2048]
            # 3. [2, 6, 2048] -> [2*6=12, 2048]
            sample = sample.reshape(-1, sample.shape[-1])  # [5*6=30, 2048]
        
        return sample

    def process_wifi(self, sample):
        # WiFi数据处理
        # [2048, 3, 3, 30]
        sample = sample.permute(1, 2, 3, 0)  # [3, 3, 30, 2048]
        sample = sample.reshape(-1, sample.shape[-1])  # [3*3*30, 2048]
        
        # 全局归一化：使用序列维度的均值和标准差
        if self.normalize:
            sample = (sample - torch.tensor(self.global_mean, dtype=torch.float32)[:, None]) / \
                     (torch.tensor(self.global_std, dtype=torch.float32)[:, None] + 1e-6)
        return sample

    def process_airpods(self, sample):
        # AirPods 数据处理：处理加速度和角速度
        acceleration = sample[:, 3:6]  # 加速度: X, Y, Z (列索引 3 到 5)
        rotation = sample[:, 6:9]      # 角速度: X, Y, Z (列索引 6 到 8)
        
        # 合并加速度和角速度数据 [2048, 6]
        sample = torch.cat((acceleration, rotation), dim=1)  # [2048, 6]
        
        # 转置为 [6, 2048]
        sample = sample.T  # 转置为 [6, 2048]
        
        # 全局归一化：使用序列维度的均值和标准差
        if self.normalize:
            sample = (sample - torch.tensor(self.global_mean, dtype=torch.float32)[:, None]) / \
                     (torch.tensor(self.global_std, dtype=torch.float32)[:, None] + 1e-6)
        return sample

def detection_collate(batch):
    """
    自定义的 collate_fn 函数，用于将目标检测任务中的样本组合成一个批次。

    参数:
        batch (list): 一个包含多个样本的列表，每个样本是一个元组 (data_dict, target)。
                      - data_dict (dict): 包含多模态数据的字典，每个键对应一种模态的数据。
                      - target (torch.Tensor): 目标框和标签信息。

    返回:
        clips (dict): 包含堆叠后的多模态数据的字典，每个键对应一种模态的批次数据。
        targets (list): 包含目标框和标签信息的列表，每个元素是一个张量。
    """
    # 初始化一个字典 clips，用于存储每种模态的数据
    # batch[0][0] 是第一个样本的 data_dict，提取其键作为 clips 的键
    clips = {key: [] for key in batch[0][0].keys()}
    

    # 初始化一个列表 targets，用于存储目标框和标签信息
    targets = []

    # 遍历 batch 中的每个样本
    for sample in batch:
        data_dict = sample[0]  # 提取样本的数据字典（多模态数据）
        target = sample[1]     # 提取样本的目标框和标签信息

        # 将每种模态的数据分别添加到 clips 中对应的键
        for key in data_dict.keys():
            clips[key].append(data_dict[key])

        # 将目标框的类别部分（最后一列）转换为整数类型
        target[:, -1] = target[:, -1].to(torch.long)
        # 将处理后的目标框和标签添加到 targets 列表中
        targets.append(target)

    # 将 clips 中每种模态的数据堆叠成一个批次
    for key in clips.keys():
        clips[key] = torch.stack(clips[key], 0)

    # 返回堆叠后的多模态数据和目标框/标签信息
    return clips, targets