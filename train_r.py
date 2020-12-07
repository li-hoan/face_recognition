#训练R网络
import net as net
import train as train

if __name__ == '__main__':
    net = net.RNet()

    trainer = train.Trainer(net, './param/rrnet.pt', r'D:\data_image\tset_result\24')# 网络，保存参数，训练数据；创建训练器
    trainer.train()