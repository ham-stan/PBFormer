from sklearn.neighbors import KDTree
import numpy as np
import pickle
from helper_ply import read_ply

f_n = 'pointcloud_1_t2'
root = 'data/03/shuffled/' + f_n + '.ply'

data = read_ply(root)
sub_xyz = np.stack((data['x'], data['y'], data['z']), axis=1)


search_tree = KDTree(sub_xyz, leaf_size=50)
kdt_name = 'data/03/shuffled/' + f_n + '_KDTree.pkl'
with open(kdt_name, 'wb') as f:
    pickle.dump(search_tree, f)
