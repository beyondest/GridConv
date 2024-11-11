import numpy as np

def semantic_grid_trans(src_graph_pose): # (1, 17, 2)
    assert len(src_graph_pose.shape) == 3 # B*J*C
    batch_size, _, C = src_graph_pose.shape
    grid_pose = np.zeros([batch_size, 5, 5, C])
    grid_pose[:, 0] = src_graph_pose[:, [7, 7, 7, 7, 7]] # (1, 5, 2)
    grid_pose[:, 1] = src_graph_pose[:, [0, 8, 8, 8, 0]] # (1, 5, 2)
    grid_pose[:, 2] = src_graph_pose[:, [1, 14, 0, 11, 4]]
    grid_pose[:, 2, 2] = src_graph_pose[:, [8, 9]].mean(1)  # midpoint of neck and nose

    grid_pose[:, 3] = src_graph_pose[:, [2, 15, 9, 12, 5]]
    grid_pose[:, 4] = src_graph_pose[:, [3, 16, 10, 13, 6]]

    grid_pose = grid_pose.transpose([0, 3, 1, 2])   # B*C*5*5

    return grid_pose
def inverse_semantic_grid_trans(src_grid_pose):
    batch_size, C = src_grid_pose.shape[:2]
    src_grid_pose = src_grid_pose.transpose([0, 2, 3, 1])  # B*5*5*C

    graph_pose = np.zeros([batch_size, 17, C])
    graph_pose[:, 7] = src_grid_pose[:, 0].mean(axis=1)
    graph_pose[:, 0] = src_grid_pose[:, 1, [0, 4]].mean(axis=1)
    graph_pose[:, 8] = src_grid_pose[:, 1, [1, 2, 3]].mean(axis=1)
    graph_pose[:, [1, 14, 11, 4]] = src_grid_pose[:, 2, [0, 1, 3, 4]]
    graph_pose[:, [2, 15, 9, 12, 5]] = src_grid_pose[:, 3]
    graph_pose[:, [3, 16, 10, 13, 6]] = src_grid_pose[:, 4]


    return graph_pose
if __name__ == '__main__':
    
    values = np.array([[i, i] for i in range(1, 18)])
    output = np.random.rand(1, 3, 5, 5)
    result_array = values.reshape(1, 17, 2)
    grid_pose = semantic_grid_trans(result_array)
    print(grid_pose.shape)
    output = inverse_semantic_grid_trans(output)
    print(output.shape)
    
    
    
    