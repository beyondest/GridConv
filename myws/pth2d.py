import torch
datapath = 'd:/VS_ws/python/GridConv/src/data/gt/test_custom_2d_unnorm.pth.tar'
data = torch.load(datapath,encoding = 'latin1')
print(data.keys())
sample = next(iter(data.values()))
print(len(sample))
s = data[('S11', 'Walking', 'Walking.60457274')]
print(s.shape)

