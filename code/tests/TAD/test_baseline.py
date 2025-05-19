import sys
sys.path.append('/home/yanrui/code/CLIPBased_TAD')
sys.path.append('/home/yanrui/code/WiFi_TAD/WiFiTAD_main')

import torch
from configs.config import config 
from TAD.model.tad_model import wifitad
from dataset.test.wwadl_muti_all_test import WWADLDatasetTestMutiALL
from tests.tester import Tester

import os 
os.environ['CUDA_VISIBLE_DEVICES']='0' 

if __name__ == '__main__':
    #------import data
    test_dataset = WWADLDatasetTestMutiALL(config)
    #------import model
    model = wifitad(270)
    #------train
    tester = Tester(config, test_dataset, model)
    tester.testing()

