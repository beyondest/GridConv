import torch
datapath = 'd:/Datasets/h36m/gt/test_custom_3d_unnorm.pth.tar'
data = torch.load(datapath,encoding = 'latin1')
print(data.keys())
sample = next(iter(data.values()))
print(len(sample))

S_11_walking = data[('S11', 'Walking', 'Walking.60457274')]
print(type(S_11_walking))

joint_3d = S_11_walking['joint_3d']
pelvis = S_11_walking['pelvis']
camera = S_11_walking['camera']
print(f"pelvis : {pelvis.shape}")
print(f"camera : {camera}")
print(f"joint_3d : {joint_3d.shape}")

# pelvis : (1621, 3)
# camera : {'fx': 1145.511338, 'fy': 1144.773928, 'cy': 501.882019, 'cx': 514.968197}
# joint_3d : (1621, 51)

