import numpy as np


dataset_path = "D:/Datasets/s-agcn/data/data_3d_h36m.npz"
dataset = np.load(dataset_path,allow_pickle=True)
print(dataset.keys())

ps3d = dataset['positions_3d']
print(ps3d.shape)
print(type(ps3d))
ps3d = ps3d.item()
print(ps3d.keys())


print(ps3d['S1'].keys())
walking_positions = ps3d['S1']['Walking 1']
print(walking_positions.shape)

#(3476, 32, 3)
