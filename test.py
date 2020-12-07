from PIL import Image
import os,torch
import json
import numpy as np
from torchvision import transforms
from detecter import Detector

# img = Image.open(r"D:\data_image\test\timg.jpg")
# # img.show()
# box = (100,200,500,600)
# ng = img.crop(box)
# ng.save(r"D:\data_image\test\1.jpg")
# ng.show()

# data_path = r"D:\data_image\face_r_data"
# a = os.listdir(data_path)
# # print(a)
# users = []
# files = []
# for i in a:
#     # print(i)
#     user,_ = os.path.splitext(i)
#     # print(user)
#     file= os.path.join(data_path,i)
#     # print(_)
#     users.append(user)
#     files.append(file)
#     # print(users)
#     # print(files)
# for j in files:
#     with open(j,"r") as f:
#         b = f.readlines()
#         print(b)
#         # c = b.("\n")
#         # print(c)
#         # d = float(c)
#         # print(c)
#         # print(c.shape)
#         # b=b[0]
#         # c = json.loads(b)
#         # print(c)

# a = torch.load(r"D:\deeplearing\face_decter\mtcnn\fdata\lha\19.pt")
# print(a)
# a = Image.open('cat.jpg')
# a.show()
x = torch.randn(3,416,416)
b = transforms.ToPILImage()
I = b(x)
I.show()
de = Detector()
box = de.detect(I)
print(box)
