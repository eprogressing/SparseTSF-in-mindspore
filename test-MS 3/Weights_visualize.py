import torch
from models.SparseTSF import Model as SparseTSF
import numpy as np
import torch.nn as nn
import os
import matplotlib.pyplot as plt

class config:
    def __init__(self):
        self.seq_len = 96
        self.pred_len = 96
        self.enc_in = 7
        self.period_len = 24
        self.d_model = 128
        self.model_type = 'linear'
configs = config()
path = "./checkpoints/ETTh1_96_96_SparseTSF_ETTh1_ftM_sl96_pl96_linear_test_0_seed2023/checkpoint.pth"
model = SparseTSF(configs)
checkpoint = torch.load(path)
model.load_state_dict(checkpoint)
eye_matrix = torch.eye(96,96)
expanded_matrix = eye_matrix.unsqueeze(-1)
final_tensor = expanded_matrix.repeat(1,1, 7)
output = model(final_tensor)
output = output.mean(dim = 2)
weights_list = output.detach().numpy()
weights_min = np.min(weights_list)
weights_max = np.max(weights_list)
weights_list = (weights_list - weights_min) / (weights_max - weights_min)
save_root = ''
if os.path.exists('weights_plot'):
    os.mkdir('weights_plot')
    
fig,ax = plt.subplots()
w_name = 'SparseTSF'
im = ax.imshow(weights_list,cmap='plasma_r')
fig.colorbar(im,pad = 0.1)
plt.savefig(os.path.join(save_root,w_name+'.pdf'),dpi=500)
plt.close
print(output)
