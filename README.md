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


