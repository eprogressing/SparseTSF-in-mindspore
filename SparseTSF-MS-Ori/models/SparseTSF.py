import mindspore 
import mindspore.nn as nn
from mindspore import ops

class SparseTSF(nn.Cell):
    def __init__(self, configs):
        super(SparseTSF, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # 自定义卷积操作
        self.conv1d = nn.SequentialCell([
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                      stride=1, pad_mode="pad", padding=self.period_len // 2, has_bias=False)
        ])

        # 自定义全连接操作
        self.linear = nn.Dense(self.seg_num_x, self.seg_num_y, has_bias=False)


    def construct(self, x):
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = ops.mean(x, axis=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # sparse forecasting
        y = self.linear(x)  # bc,w,m

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean
        
        #print("[State-SparseTSF,enc_in]:{0}".format(self.enc_in))

        return y
