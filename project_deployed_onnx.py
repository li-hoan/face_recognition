import io
from PIL import Image
import net
import torch
from torch import nn
from torchvision import transforms
from detecter import Detector



class kaung(nn.Module):
    def __init__(self):
        super().__init__()
        self.net =Detector()
        self.tf = transforms.ToPILImage()
    def forward(self,x):
        img = self.tf(x[0])
        boxes = self.net.detect(img)
        return boxes

model = kaung()
print(model)
batch_size=1

x = torch.randn(batch_size,3,416,416,requires_grad=False)
out = model(x,)
# print(out)

#打包
torch.onnx.export(model,
                  x,
                  "kuang_face.onnx",
                  export_params=False,

                  verbose=False,
                  opset_version=10,
                  do_constant_folding=True,
                  training=False,
                  input_names=['input'],
                  output_names=['output'],
                 )