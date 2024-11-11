import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.tool.argument import Options
from src.base_modules import get_lifting_model



weight_path = 'D:\VS_ws\python\GridConv\src\pretrained_model\gt_d-gridconv_autogrids.pth.tar'


if __name__ == '__main__':
    opt = Options().parse()
    opt.lifting_model = 'dgridconv_autogrids'
    opt.eval = True
    opt.input = 'gt'
    opt.load = weight_path
    opt.padding_mode = ['c', 'r']
    opt.prepare_grid = False
    model = get_lifting_model(opt)
    model.eval()
    dummy_input = torch.rand(1, 17, 2)
    dummy_output = torch.rand(1, 17, 3)
    output = model(dummy_input)
    print(output.shape) 
    print(output)
    
