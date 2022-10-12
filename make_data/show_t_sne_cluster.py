from typing import Pattern
import numpy as np
import os
import shutil
from t_sne import tsne
from glob import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def main():
    cluster_data_path_ls = glob(os.path.join(source_dir, pattern))
    for cluster_data_path in cluster_data_path_ls:
        print(cluster_data_path)
        cluster_data_name = os.path.basename(cluster_data_path).split('.')[0]
        cluster_data = np.load(cluster_data_path)[::down_sample_rate]  
        
        Y = tsne(cluster_data, 2, 50, 30.0)
        plt.scatter(Y[:,0], Y[:,1], 20) 
        cluster_save_path = os.path.join(save_dir, cluster_data_name + '.jpg')
        plt.savefig(cluster_save_path)
        plt.close()
        del cluster_data, Y

if __name__ == '__main__':
    source_dir = '/home/yaboliu/work/research/gepc/gepc_0/data/exp_dir/stc/Jan11_0359/checkpoints'
    save_dir = os.path.join(source_dir, 'cluster_vis')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    pattern = 'cluster_*_epoch23.npy'
    down_sample_rate = 500
    main()

