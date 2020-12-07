#构建P/R/O网络
import torch.nn.functional as F
from torch import nn
import torch


class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3,10,3,1,padding=1),
            nn.MaxPool2d(3,2),
            nn.PReLU(),

            nn.Conv2d(10,16,3),
            nn.PReLU(),

            nn.Conv2d(16,32,3),
            nn.PReLU()
        )
        self.comv4_1 = nn.Conv2d(32,1,1)
        self.comv4_2 = nn.Conv2d(32,4,1)

    def forward(self,x):
        x = self.pre_layer(x)
        cond = torch.sigmoid(self.comv4_1(x))
        offset = self.comv4_2(x)
        return cond,offset
#r网络
class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3,28,3,padding=1),
            nn.MaxPool2d(3,2),
            nn.PReLU(),

            nn.Conv2d(28,48,3),
            nn.MaxPool2d(3,2),
            nn.PReLU(),

            nn.Conv2d(48,64,2),
            nn.PReLU()
        )
        self.conv4 = nn.Linear(64*3*3,128)
        self.prelu4 = nn.PReLU()

        self.conv5_1 = nn.Linear(128,1)
        self.conv5_2 = nn.Linear(128,4)

    def forward(self,x):
        x = self.pre_layer(x)
        x = x.reshape(x.size(0),-1)
        x=  self.conv4(x)
        x = self.prelu4(x)

        label  =torch.sigmoid(self.conv5_1(x))
        offset =self.conv5_2(x)
        return label,offset

class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.MaxPool2d(3,2),
            nn.PReLU(),

            nn.Conv2d(32,64,3),
            nn.MaxPool2d(3,2),
            nn.PReLU(),

            nn.Conv2d(64,64,3),
            nn.MaxPool2d(2,2),
            nn.PReLU(),

            nn.Conv2d(64,128,2),
            nn.PReLU()
        )

        self.conv5 = nn.Linear(128*3*3,256)
        self.prelu5 = nn.PReLU()
        self.conv6_1 = nn.Linear(256,1)
        self.conv6_2 = nn.Linear(256,4)

    def forward(self,x):
        x = self.pre_layer(x)
        x = x.reshape(x.size(0),-1)
        x = self.conv5(x)
        x = self.prelu5(x)

        label  = torch.sigmoid(self.conv6_1(x))
        offset = self.conv6_2(x)
        return label,offset

if __name__ == '__main__':
    net1 = PNet()
    net2 = RNet()
    net3 = ONet()

    weight1 = r"D:\deeplearing\face_decter\mtcnn\param\rnet.pt"
    weight2 = r"D:\deeplearing\face_decter\mtcnn\param\rrnet.pt"
    weight3 = r"D:\deeplearing\face_decter\mtcnn\param\onet.pt"

    batch_size = 1
    w = 416
    h = 416

    #调用权重
    net1.load_state_dict(torch.load(weight1))
    net2.load_state_dict(torch.load(weight2))
    net3.load_state_dict(torch.load(weight3))

    net1.eval()
    net2.eval()
    net3.eval()

    x1 = torch.randn(batch_size,3,h,w,requires_grad=True)
    x2 = torch.randn(batch_size,3,24,24,requires_grad=True)
    x3 = torch.randn(batch_size,3,48,48,requires_grad=True)


    label1,offset1 = net1(x1)
    label2, offset2 = net2(x2)
    label3, offset3 = net3(x3)

    torch.onnx.export(net1,
                      x1,
                      "FILE/Pnet.onnx",
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input1'],
                      output_names=['output1'],
                      dynamic_axes={'input1':{0:'batch_size',2:'h',3:'w'}}
                      )


    torch.onnx.export(net2,
                      x2,
                      "FILE/Rnet.onnx",
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input2'],
                      output_names=['output2'],
                      dynamic_axes={'input2': {0: 'batch_size', 2: 'h', 3: 'w'}}
                      )

    torch.onnx.export(net3,
                      x3,
                      "FILE/Onet.onnx",
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input3'],
                      output_names=['output3'],
                      dynamic_axes={'input3': {0: 'batch_size', 2: 'h', 3: 'w'}}
                      )

    import onnx
    import numpy as np

    onnx_model = onnx.load("FILE/Pnet.onnx")
    onnx.checker.check_model(onnx_model)

    import onnxruntime

    ort_session = onnxruntime.InferenceSession("FILE/Pnet.onnx")


    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    # compute ONNX Runtime output prediction
    print(x1.shape)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x1)}
    label1_or,offset1_or = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(label1[0]), label1_or[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
