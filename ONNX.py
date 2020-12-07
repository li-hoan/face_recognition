import onnx
import torch
from detecter import Detector
from facenet import Net
from torch import nn
from torchvision import transforms
from  f_da_control import CON
from PIL import Image
import cv2
import numpy as np

class FacerelizeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = []
        detecetr = Detector()
        self.detector = detecetr
        self.facenet = Net()
        self.facenet.load_state_dict(torch.load(r"D:\deeplearing\face_decter\mtcnn\parms\arc_100.pt"))
        self.con = CON()
    def forward(self,x):
        x = np.array(x)
        img = Image.fromarray(x[:,:,::-1],"RGB")
        boexs = self.detector.detect(img)
        ku =  self.__fa(boexs,img)
        return ku
    def __fa(self,boxes,img):
        fan = []
        for i ,box in enumerate(boxes):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            w = x2-x1
            h = y2-y1
            a = max(w,h)

            b = a-w
            c = a-h

            ng = img.crop((x1+1, y1+1, x2+b-1, y2+c-1))
            ng_t = ng.resize((100,100))
            ng_t.show()
            tf  = transforms.ToTensor()
            ng_data = tf(ng_t)
            ng_data = torch.unsqueeze(ng_data, dim=0)
            print(ng_data.shape)
            ng_feat, _ = self.facenet(ng_data)
            name, cos = self.con(ng_feat)
            fan.append((x1,y1,x2,y2,name,cos))
        return fan

fanet = FacerelizeNet()
#输入模型
x = cv2.imread("2.jpg")
x = torch.from_numpy(x)
out  = fanet(x)
print(out)

#打包模型
torch.onnx.export(fanet,
                  x,
                  "facerelize.onnx",
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output']
                  )