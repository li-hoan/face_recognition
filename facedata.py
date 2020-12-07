#为特征提取器，做数据准备
from torch.utils.data import Dataset
import torch
import os
from torchvision import transforms
from PIL import Image

class Facedata(Dataset):
    def __init__(self,root,is_train=True):
        super().__init__()
        self.dataset = []
        sub = "TRAIN" if is_train else "TEST"
        for tag in os.listdir(f"{root}/{sub}"):
            img_tag = f"{root}/{sub}/{tag}"
            for img in os.listdir(img_tag):
                img_path = f"{img_tag}/{img}"
                self.dataset.append((img_path,tag))
    def __len__(self):
        return  len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        img = Image.open(data[0])
        img = img.resize((100,100))
        tag = data[1]
        tf = transforms.ToTensor()
        img_data = tf(img)
        return img_data,int(tag)


if __name__ == '__main__':
    dataset = Facedata(r"D:\data_image\facedata")
    print(dataset[20])
    print(type(dataset[20][1]))
    print(type(dataset[20][0]))
    print(dataset[10][0].shape)




