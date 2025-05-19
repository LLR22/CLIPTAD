import sys
sys.path.append('/home/yanrui/code/CLIPBased_TAD')
sys.path.append('/home/yanrui/code/WiFi_TAD/WiFiTAD_main')


import torch
from configs.config import config 
from model.CLIPTAD.CLIPTAD import CLIPTAD
from model.models import RFCLIP, XRFMamba
from dataset.train.wwadl_multi import WWADLDatasetMutiAll
from trains.train import Trainer

import os 
os.environ['CUDA_VISIBLE_DEVICES']='3' 

if __name__ == '__main__':
    #------import data
    train_dataset_multi = WWADLDatasetMutiAll(config['path']['train_data_path'], split='train')
    #------import model
    model = RFCLIP(config)
    # model = XRFMamba(config)
    path_checkpoint = '/home/yanrui/code/CLIPBased_TAD/result/CLIPMamba_CNN_TADdeep_GatefusionAdd-epoch-790.pt'
    checkpoint = torch.load(path_checkpoint, map_location = torch.device('cpu'))
    model.load_state_dict(checkpoint,strict=True)
    model.eval()  # 先设为评估模式
    #------train
    trainer = Trainer(config, train_dataset_multi, model)
    trainer.training()