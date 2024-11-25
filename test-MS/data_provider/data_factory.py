from data_provider.data_loader import Dataset_ETT_hour
from mindspore.dataset import GeneratorDataset
import mindspore.dataset as ds

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    #'ETTm1': Dataset_ETT_minute,
    #'ETTm2': Dataset_ETT_minute,
    #'custom': Dataset_Custom,
}

def data_provider(args,flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed!='timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False  # fix bug
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    # 提醒 我们这里并没有写Predict，因为暂时实验不涉及Predict，最后我们有时间的话补充

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

    data_loader = GeneratorDataset(source=data_set, column_names=["seq_x", "seq_y", "seq_x_mark", "seq_y_mark"],shuffle = shuffle_flag) #修改建模为Mindspore的产生源数据的方式
    
    print(flag, data_loader.get_dataset_size()) #这里的函数等价于print(flag, len(data_set))

    ## 有稍微不一样的地方就是没有batch和drop——last
    data_loader = data_loader.batch(batch_size = batch_size, drop_remainder=drop_last)

    return data_set, data_loader




