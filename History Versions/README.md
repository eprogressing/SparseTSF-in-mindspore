
+ **v1.0**: SparseTSF能够正常地在MindSpore框架下运行，但是存在说过早早停（过拟合的现象），并且莫名其妙地，我们MSE很低

Solution：
1.`` enc-in = 1``的粗心

2. 发现``batch-y``, ``outputs``,以及``batch-x``的形状异常，发现是因为``feature``的模式错误导致传入``f-dim = -1``，但是正确的``f-dim = 0``

3. 没有加入``adjust-learning-rate``的模块

+ **v2.0**: SparseTSF只能在Etth1，2上面正常运行并且正常反馈，流程正常，MSE几乎和在Torch框架下保持一致。但是早停的步调和Torch框架下不一致

Solution：我们猜测是因为梯度“反向传播”参数的机制导致的问题。
