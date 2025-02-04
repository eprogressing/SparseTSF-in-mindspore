from mindspore import numpy as ms_np
import mindspore.ops as ops
from mindspore.ops.function import broadcast_to
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore import ms_function

class TriangularCausalMask():
    def __init__(self, B, L):
        mask_shape = (B, 1, L, L)
        self._mask = ops.stop_gradient(nn.Triu()(ops.ones(mask_shape, type=mstype.bool_), k=1))

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores):
        _mask = nn.Triu()(ops.ones((L, scores.shape[-1]), type=mstype.bool_), k=1)
        _mask_ex = broadcast_to(_mask[None, None, :], (B, H, L, scores.shape[-1]))
        indicator = _mask_ex[ms_np.arange(B)[:, None, None],
                             ms_np.arange(H)[None, :, None],
                             index, :]
        self._mask = indicator.view(scores.shape)
    
    @property
    def mask(self):
        return self._mask

class ProbMaskCell(nn.Cell):
    def __init__(self):
        super(ProbMaskCell, self).__init__()
    
    def construct(self, B, H, L, index, scores):
        _mask = nn.Triu()(ops.ones((L, scores.shape[-1]), type=mstype.bool_), k=1)
        _mask_ex = broadcast_to(_mask[None, None, :], (B, H, L, scores.shape[-1]))
        indicator = _mask_ex[ms_np.arange(B)[:, None, None],
                             ms_np.arange(H)[None, :, None],
                             index, :]
        return indicator.view(scores.shape)