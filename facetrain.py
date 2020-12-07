#训练网络特征提取器
import torch
from torch.utils.data import DataLoader
import torchvision
# from arf import *
from facedata import *
from facenet import *
from torchvision import models
from tqdm import tqdm

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Train:
    def __init__(self,root):
        self.train_dataset = Facedata(root)
        self.train_dataloader = DataLoader(self.train_dataset,batch_size=20,shuffle=True)

        self.test_dataset = Facedata(root,is_train=False)
        self.test_dataloader = DataLoader(self.test_dataset,batch_size=10,shuffle=True)

        self.net = Net()
        # self.arc_net = Arc()
        self.net.to(Device)
        # self.arc_net.to(Device)

        self.opt_net = torch.optim.Adam(self.net.parameters())
        # self.opt_arc = torch.optim.Adam(self.arc_net.parameters())

        self.loss_fun = nn.NLLLoss()

    def __call__(self):
        for epoch in range(1000):
            print("目前轮次为：",epoch)
            loss_sum = 0.
            for i,(data,tag) in enumerate(tqdm(self.train_dataloader)):
                data,tag = data.to(Device),tag.to(Device)
                feature,cls = self.net(data)
                loss = self.loss_fun(cls,tag)

                self.opt_net.zero_grad()
                # self.opt_arc.zero_grad()
                loss.backward()
                self.opt_net.step()
                # self.opt_arc.step()

                loss_sum+=loss.cpu().detach().item()
            avg_loss = loss_sum/len(self.train_dataloader)

            #测试
            for i,(img,tag) in enumerate(self.test_dataloader):
                img,tag = img.to(Device),tag.to(Device)
                test_f,test_t = self.net(img)
                # test_out = self.arc_net(test_y)
                test_loss = self.loss_fun(test_t,tag)
                pre_tag = torch.argmax(test_t,dim=1)
                a = torch.sum(torch.eq(pre_tag,tag).float())#/len(pre_tag)
            print(epoch,"train_loss:",avg_loss,"test_loss:",test_loss.item(),a)
            torch.save(self.net.state_dict(),r'.\parms\arc_{}.pt'.format(epoch))
            # torch.save(self.arc_net, r'.\parms\net_{}.pt'.format(epoch))
            print('save susccesfully')
if __name__ == '__main__':
    train = Train(r"D:\data_image\facedata")
    train()




