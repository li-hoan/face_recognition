#为特征对比，做数据集准备
from torch.utils.data import Dataset
import torch
import os

class FAdata(Dataset):
    def __init__(self,root=r"D:\deeplearing\face_decter\mtcnn\fdata"):
        super().__init__()
        self.dataset = []
        self.taget =0
        self.tagets = []
        for filename in os.listdir(root):
            name = filename
            for fna in os.listdir(f"{root}/{filename}"):
                path = f"{root}/{filename}/{fna}"
                feature_tensor = torch.load(path)
                self.dataset.append((self.taget,feature_tensor))
            # print(self.dataset)
            self.tagets.append(name)
            self.taget += 1
    def __len__(self):
        return  len(self.dataset)

    def __getitem__(self, index):
        target,feature = self.dataset[index]
        return feature,torch.tensor(target)
if __name__ == '__main__':
    net =FAdata()
    c,d  =net[0]
    print(c,d)



