#特征对比
from f_da_dataset import FAdata
from torch.utils.data import DataLoader
import torch

class CON():
    def __init__(self):
        self.dataset = FAdata()
        self.dataloader = DataLoader(self.dataset,len(self.dataset),shuffle=False)
    def __call__(self,feat):
        tagets = self.dataset.tagets
        information = []
        for feature,target in self.dataloader:
            cos_thetas = torch.cosine_similarity(feat,feature,dim=1)
            ma_index = torch.argmax(cos_thetas)

            cos_theta = cos_thetas[ma_index]
            print(cos_theta.item())
            if cos_theta >=0.90:
                name = tagets[target[ma_index.item()]]
                return name,cos_theta



