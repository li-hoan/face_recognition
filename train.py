#mtcnn创建训练器

import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from data import FaceDataset
import net as net


class Trainer:
    def __init__(self,net,save_path,dataset_path,isCuda=True):
        self.net = net
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.isCuda = isCuda



        #创建优化器
        self.opt = optim.Adam(self.net.parameters())

        #恢复网络训练--加载模型参数，继续训练
        if os.path.exists(self.save_path):#如果文件存在，接着继续训练
            net.load_state_dict(torch.load(self.save_path))


    #训练方法
    def train(self):
        facedataset = FaceDataset(self.dataset_path)
        dataloader = DataLoader(facedataset,batch_size=512,shuffle=True,num_workers=4,drop_last=True)
        a=0
        #drop_last:为True表示防止批次不足报错。去掉最后一批


        while True:
            a = a+1
            for i,(img_data_,category_,offset_) in enumerate(dataloader):#样本，置信度，偏移量
                if self.isCuda:
                    img_data_ = img_data_.cuda()#[512,3,12,12]
                    category_ = category_.cuda()#[512,1]
                    offset_ = offset_.cuda()#[512,4]

                #网络输出
                _output_category,_output_offset = self.net(img_data_)#输出置信度，偏移量
                output_category = _output_category.reshape(-1,1)#[512,1]
                output_offset = _output_offset.reshape(-1,4)#[512,4]

                #计算分类的损失---置信度
                category_mask = torch.lt(category_,2)#对于置信度小于2的正样本1和0
                category = torch.masked_select(category_,category_mask)
                output_category = torch.masked_select(output_category,category_mask)
                cls_loss = self.cls_loss_fn(output_category,category)

                #计算回归的的损失-偏移量
                offset_mask = torch.gt(category_,0)#对于置信度大于0的标签
                offset_index = torch.nonzero(offset_mask)[:,0]#获取非0元素的索引值，因为维度原因后面加【：，0】索引
                offset = offset_[offset_index]
                output_offset = output_offset[offset_index]
                offset_loss = self.offset_loss_fn(output_offset,offset)#偏移量损失


                #总损失
                loss = cls_loss+offset_loss

                #优化
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


            print("i=",a,"loss",loss.cpu().data.numpy(),"cls_loss:",cls_loss.cpu().data.numpy(),"offset_loss:",offset_loss.cpu().data.numpy())

            #保存

            torch.save(self.net.state_dict(),self.save_path)
            print("save success")
# if __name__ == '__main__':
#     net = net.ONet()
#
#     trainer = Trainer(net, './param/onet.pt', r'D:\data_image\tset_result\48')  # 网络，保存参数，训练数据；创建训练器
#     trainer.train()















