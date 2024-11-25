from exp.exp_basic_ms import Exp_Basic
from models.SparseTSF import SparseTSF
from data_provider.data_factory import data_provider

from utils.tools_ms import EarlyStopping,adjust_learning_rate
from utils.metrics_ms import metric
import os
import mindspore.nn as nn
import numpy as np
import mindspore.ops as ops
import mindspore as ms
import time
import matplotlib.pyplot as plt

class Exp_SparseTSF(Exp_Basic):
    def __init__(self,args):
        super(Exp_SparseTSF,self).__init__(args)
    def _build_model(self):
        model_dict = {
            'SparseTSF':SparseTSF
        }
        model = model_dict[self.args.model](self.args)
        return model
    
    def _get_model(self):
        return self.model
    
    def _get_data(self,flag):
        data_set,data_source= data_provider(self.args,flag)
        return data_set,data_source
    def _select_optimizer(self):
        model_optim = nn.Adam(self.model.trainable_params(),learning_rate=self.args.learning_rate)
        print("Matrix Format:\n{0}".format(self.model.trainable_params()[1]))
        return model_optim
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def vali(self,model,vali_data,vali_source,criterion):
        total_loss = []
        model.set_train(False)
        for i,(batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_source.create_tuple_iterator()):
            pred,ture = self._process_one_batch(
                model,vali_data,batch_x,batch_y,batch_x_mark,batch_y_mark
            )
            
            pred_detached = ops.stop_gradient(pred)
            ture_detached = ops.stop_gradient(ture)
            loss = criterion(pred_detached,ture_detached).asnumpy()   
            total_loss.append(loss)
            
        train_loss = np.average(total_loss)
        model.set_train()
        return train_loss

    def train(self,model,setting):
        train_data,train_source = self._get_data(flag = 'train')
        vali_data,vali_source = self._get_data(flag = 'val')
        test_data,test_source = self._get_data(flag = 'test')
    
        path = os.path.join(self.args.checkpoints,setting)
        if not os.path.exists(path):
            os.makedirs(path)
        train_steps = train_source.get_dataset_size()
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        
        
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
        time_now = time.time()
        model.set_train()
        
        train_losses = []
        
        early_stopping = EarlyStopping(patience = self.args.patience,verbose = True)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for i,(batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_source.create_tuple_iterator()):
                iter_count += 1
                grad_fn = ms.ops.value_and_grad(forward_fn,None,model_optim.parameters,has_aux=True)
                (loss,_),grads = grad_fn(batch_x,batch_y,batch_x_mark,batch_y_mark,self.args.label_len,self.args.pred_len)
                loss = ms.ops.depend(loss,model_optim(grads))
                train_loss.append(loss.asnumpy().item())
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.asnumpy().item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    
                
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            train_losses.append(train_loss)#保存
            
            vali_loss = self.vali(model, vali_data, vali_source, criterion)
            test_loss = self.vali(model, test_data, test_source, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            # 添加学习率修改 因为出现了过早早停的现象
            adjust_learning_rate(model_optim, self.model.trainable_params(), epoch+1, self.args)
        best_model_path = path+'/'+'checkpoint.ckpt'
        ms.load_param_into_net(model, ms.load_checkpoint(best_model_path))
        
        Tepochs = range(1, len(train_losses) + 1)  # 每一轮的编号

        # 绘制训练损失曲线
        plt.plot(Tepochs, train_losses, label='Training Loss', color='blue', marker='o')

        # 添加标签、标题和图例
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid()  # 可选：添加网格线

        # 显示图像
        plt.savefig('train_loss_curve.png')
        
    
    def test(self,model,setting):
        test_data,test_source = self._get_data(flag = 'test')
        model.set_train(False)
        
        preds = []
        trues = []
        
        for i,(batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_source.create_tuple_iterator()):
            pred,true = self._process_one_batch(
                model,test_data,batch_x,batch_y,batch_x_mark,batch_y_mark
            )
            preds.append(pred.asnumpy())
            trues.append(true.asnumpy())
        model.set_train()
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        preds = preds.reshape(-1,preds.shape[-2],preds.shape[-1])
        trues = trues.reshape(-1,trues.shape[-2],trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        return
    #处理单批次
    def _process_one_batch(self,model,dataset_object,batch_x,batch_y,batch_x_mark,batch_y_mark):
        cast = ops.Cast()
        batch_x = cast(batch_x,ms.float32)
        batch_y = cast(batch_y,ms.float32)
        batch_x_mark = cast(batch_x_mark,ms.float32)
        batch_y_mark = cast(batch_y_mark,ms.float32)
        outputs = model(batch_x)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:,-self.args.pred_len:,f_dim:]
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:]
        #print("Outputs:{0},Batch_y:{1}".format(outputs.shape,batch_y.shape))
        return outputs,batch_y        