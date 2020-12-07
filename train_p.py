#训练P网络
import net as net
import train as train
if __name__ == '__main__':
    net = net.PNet()

    trainer = train.Trainer(net, './param/rnet.pt', r'D:\data_image\tset_result\12')# 网络，保存参数，训练数据；创建训练器
    trainer.train()