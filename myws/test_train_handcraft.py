import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.tool.argument import Options
from src.base_modules import get_lifting_model

from src.main import main

dataset_path = "D:/Datasets/h36m"

if __name__ == '__main__':
    opt = Options().parse()
    opt.lifting_model = 'dgridconv'
    opt.eval = False
    opt.input = 'gt'
    opt.padding_mode = ['c', 'r']
    opt.data_rootdir = dataset_path
    opt.prepare_grid = True
    opt.exp = 'dgridconv_weights'
    
    main(opt)
    

    
    
