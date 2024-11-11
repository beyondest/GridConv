import numpy as np


dataset_path = "D:/Datasets/s-agcn/data/data_2d_h36m_gt.npz"
dataset = np.load(dataset_path,allow_pickle=True)
print(dataset.keys())

ps2d = dataset['positions_2d']
print(ps2d.shape)
print(type(ps2d))
ps2d = ps2d.item()
print(ps2d.keys())

metadata = dataset['metadata'].item()
print(metadata.keys())
print(metadata['num_joints'])
print(metadata['keypoints_symmetry'])
print(ps2d['S1'].keys())
list_directions_1 = ps2d['S1']['Directions 1']
print(len(list_directions_1))
print(list_directions_1[0].shape)

