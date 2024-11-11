import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.base_modules import get_dataloader

from tool.argument import Options



if __name__ == '__main__':
    opt = Options().parse()
    opt.lifting_model = 'dgridconv'
    opt.load = 'pretrained_model/gt_d-gridconv.pth.tar'
    opt.eval = True
    opt.input = 'gt'
    opt.padding_mode = ['c','r']
    opt.data_rootdir = 'd:/VS_ws/python/GridConv/src/data/'
    test_dataloader = get_dataloader(opt,is_train=False,shuffle=False)
    print(len(test_dataloader))
    sample = next(iter(test_dataloader))
    input_data, target_data, meta = sample
    print(input_data.shape)
    print(target_data.shape)
    print(meta.keys())
    
    