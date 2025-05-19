import sys
sys.path.append('/home/yanrui/code/CLIPBased_TAD')
sys.path.append('/home/yanrui/code/WiFi_TAD/WiFiTAD_main')

import torch
from configs.config import config 
from model.CLIPTAD.CLIPTAD import CLIPTAD
from model.models import RFCLIP, XRFMamba
from model.Two_stage_models.ClipClsLoc import ClipClsLoc
from model.Two_stage_models.ClipCls import ClipCls
from dataset.test.wwadl_muti_all_test import WWADLDatasetTestMutiALL
from tests.tester import Tester

import os 
os.environ['CUDA_VISIBLE_DEVICES']='2' 

if __name__ == '__main__':
    #------import data
    test_dataset = WWADLDatasetTestMutiALL(config)
    #------import model
    model = ClipCls(config)
    # model = XRFMamba(config)
    # model = ClipClsLoc(config)
    #------train
    tester = Tester(config, test_dataset, model)
    tester.testing()

