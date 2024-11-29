# SparseTSF-in-mindspore

SparseTSF在MindSpore上的代码迁移

## 模型启动前准备

```bash
conda create -n SparseTSF-MS python=3.8
```

然后我们启动SparseTSF-MS环境并进行相应的配置

```bash
conda activate SparseTSF-MS
pip install -r requirements.txt
```

在仓库中的数据集上的模型训练测试的指令：

```bash
python run.py --config_name configs/GPU_ETTh1.yaml
```

### 主要修改部分

#### 1.data_factory.py

由于在原来的data_factory.py文件主要作用是构建Dataset、DataLoader类用于模型的数据读取

Dataset类是所有数据集的基类，提供了数据处理方法来帮忙预处理数据

Dataloader类则是数据集上的python可迭代对象，方便我们进行批处理

下面是在torch中与mindspore中代码的对比

```pytho
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
```

```pyt
import mindspore.dataset as ds	
from mindspore.dataset import GeneratorDataset
data_set = Data(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=args.data_path,
        target=args.target,
        inverse=args.inverse, #new
        timeenc=timeenc,
        freq=freq,
        cols=args.cols #new
    )
data_loader = GeneratorDataset(source=data_set, column_names=["seq_x", "seq_y", "seq_x_mark", "seq_y_mark"],shuffle = shuffle_flag)
data_loader = data_loader.batch(batch_size = batch_size, drop_remainder=drop_last)
```

发现主要区别在于：在mindspore中，batch_size需要特别地设置

#### 2.SparseTSF的修改

首先最明显的不同在于，模型继承的类不同，torch中继承的是torch.nn.Module,而mindspore的在mindspore.nn.Cell

```py
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from layers.Embed import PositionalEmbedding

class Model(nn.Module):
```

```python
import mindspore 
import mindspore.nn as nn
from mindspore import ops

class SparseTSF(nn.Cell):
```

其次，在我们在mindspore中设计神经网络时，一方面，神经网络模块必须放入nn.SequentialCell容器，这个容器允许将多个网络层按顺序组合起来，例如我的神经网络是经过一层卷积层然后经过激活函数最后进行一次卷积的话，就往这个容器中append进自己定义的卷积、激活函数、卷积，然后将输入传入这个容器就可以自动计算出经过我们自己定义的网络的结果。其次，mindspore中的全连接层是封装在Dense模块中

```python
self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)
self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
```



```python
self.conv1d = nn.SequentialCell([
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                      stride=1, pad_mode="pad", padding=self.period_len // 2, has_bias=False)
        ])

        # 自定义全连接操作
        self.linear = nn.Dense(self.seg_num_x, self.seg_num_y, has_bias=False)
```

相较于torch中的forward函数进行前向传播操作，mindspore则是通过函数construct来进行的

```python
 def forward(self, x):
```

```python
def construct(self, x):
```

其余对张量的操作，二者大致是一致的，因此不做介绍

#### 3.Exp_Main()的修改

首先一个小的修改部分在于优化器的选择时一点小小的不同,即选取参数的不同

```python
def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
```

```python
def _select_optimizer(self):
        model_optim = nn.Adam(self.model.trainable_params(),learning_rate=self.args.learning_rate)
        return model_optim
```

其次是训练时数据的迭代的在名字上不同

```python
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
```



```python
 for i,(batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_source.create_tuple_iterator()):
```

其次在前向传播和反向传播的二者存在较大的不同

对于torch而言就是简单的计算出模型计算的结果然后计算loss值，进行反向传播，参数调整

```python
batch_x = batch_x.float().to(self.device)
batch_y = batch_y.float().to(self.device)
outputs = self.model(batch_x)
f_dim = -1 if self.args.features == 'MS' else 0
outputs = outputs[:, -self.args.pred_len:,f_dim:]
batch_y = batch_y[:, -self.args.pred_len:,f_dim:].to(self.device)
loss = criterion(outputs, batch_y)
train_loss.append(loss.item())
loss.backward()
model_optim.step()
```

而mindspore则较为复杂，需要先写一个forward_fn函数，用作计算结果和损失值

```python
def forward_fn(batch_x,batch_y,batch_x_mark,batch_y_mark,label_len,pred_len):
            cast = ops.Cast()
            batch_x = cast(batch_x,ms.float32)
            #print("Batch_x:\n{0}".format(batch_x.shape))
            batch_y = cast(batch_y,ms.float32)
            
            batch_x_mark = cast(batch_x_mark,ms.float32)
            batch_y_mark = cast(batch_y_mark,ms.float32)
            ouputs = model(batch_x)
            f_dim = 0
            ouputs = ouputs[:,-self.args.pred_len:,f_dim:]
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:]
            #print("Outputs:{0},Batch_y:{1}".format(ouputs.shape,batch_y.shape))
            loss = criterion(ouputs,batch_y)
            return loss,ouputs
```

然后定义一个grad_fn函数用于在前向传播时计算梯度并得到损失值，然后将计算出来的梯度传入优化器进行参数的更新

```python
grad_fn = ms.ops.value_and_grad(forward_fn,None,model_optim.parameters,has_aux=True)
(loss,_),grads = grad_fn(batch_x,batch_y,batch_x_mark,batch_y_mark,self.args.label_len,self.args.pred_len)
loss = ms.ops.depend(loss,model_optim(grads))
train_loss.append(loss.asnumpy().item())
```

对于学习率的调整和早停机制的修改会放在后面进行介绍

#### 4.utils.tools的修改

学习率调整策略的修改

在torch中可以直接对学习率进行修改，因为每个参数都有一个['lr']的超参数；而对于mindspore的学习率调整只能新生成一个optimizer来进行学习率的更新

```python
def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.8 ** ((epoch - 3) // 1))}
     if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if printout: print('Updating learning rate to {}'.format(lr))
```

```python
def adjust_learning_rate(optimizer, parameters, epoch, args):
    lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.8 ** ((epoch - 3) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        optimizer = ms.nn.Adam(parameters, learning_rate=lr)
        print('Updating learning rate to {}'.format(lr))
    return optimizer
```

早停策略大致一致，不同之处在于参数的存储调用的函数的名称不同

标准化的调整同样也只是名字修改层面

#### 5.run.py的调整

run.py的主要作用是参数的设置以及设备的选择

设备的选择上,mindspore是采用的context来设置，设置之后，不需要向torch一样通过to_device将数据传入GPU

```python
args = dict_to_namedtuple(args)
print("DEVICE:", args.device)
ms.set_context(device_target=args.device)
ms.set_context(mode=ms.PYNATIVE_MODE)
```

其余一致
