#创建mtcnn数据集

from torch.utils.data import Dataset
import os,torch
from PIL import Image
import numpy as np

#数据集
class   FaceDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(self.path,"positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(self.path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(self.path, "part.txt")).readlines())
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split(' ')#取一条数据，去掉他的前后字符串，再按空格分隔

        # print(strs)
        #标签：置信度加偏移量
        cond = torch.Tensor([int(strs[1])])#取出置信度,[]莫丢，否则指定的是shape
        # print(cond)
        offset = torch.Tensor([float(strs[2]),float(strs[3]),float(strs[4]),float(strs[5])])
        # print(offset)


        #样本
        img_path = os.path.join(self.path,strs[0])#图片绝对路径
        img_data = torch.Tensor(np.array(Image.open(img_path))/255-0.5)
        img_data = img_data.permute(2,0,1)#chw

        return img_data,cond,offset
if __name__ == '__main__':
    fa = FaceDataset(r'D:\data_image\t')
    fa[0]




