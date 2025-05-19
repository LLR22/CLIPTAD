import sys
sys.path.append('/home/yanrui/code/CLIPBased_TAD')
sys.path.append('/home/yanrui/code/WiFi_TAD/WiFiTAD_main')

import torch
from configs.config import config 
from TAD.model.tad_model import wifitad
from dataset.train.wwadl import WWADLDatasetSingle, detection_collate
from torch.utils.data import DataLoader
from dataset.train.wwadl_multi import WWADLDatasetMutiAll
from train import Trainer

import os 
os.environ['CUDA_VISIBLE_DEVICES']='1' 

if __name__ == '__main__':
    #------import data
    train_dataset_multi = WWADLDatasetMutiAll(config['path']['train_data_path'], split='train')
    #------import model
    model = wifitad(270)
    #------train
    trainer = Trainer(config, train_dataset_multi, model)
    trainer.training()