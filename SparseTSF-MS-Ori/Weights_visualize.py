import mindspore
from models.SparseTSF import SparseTSF
import numpy as np
import mindspore.nn as nn
from mindspore import ops
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
path = "./checkpoints/SparseTSF_ETTh2_ftM_sl96_ll48_pl96_dm24_nh512_el8_dl2_df1_at2048_fcprob_eb5_dttimeF_mxTrue_True_SparseTSF_ms/checkpoint.ckpt"
model = SparseTSF(configs)
checkpoint = mindspore.load_checkpoint(path)
mindspore.load_param_into_net(model,checkpoint)
eye_matrix = mindspore.ops.eye(96,96)
expanded_matrix = eye_matrix.unsqueeze(-1)
multiples = (1,1, 7)
final_tensor = expanded_matrix.tile(multiples)

print("final_tensor:",final_tensor.shape)

output = model(final_tensor)
output = output.mean(axis = 2)
weights_list = output.asnumpy()
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
