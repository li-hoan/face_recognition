from torch import nn
import torch
import torch.nn.functional as F

class Arc(nn.Module):
    def __init__(self,feature_dim=1000,cls_dim=3):
        super().__init__()
        self.W = nn.Parameter(torch.randn(feature_dim,cls_dim))
    def forward(self,feature,m=1,s=10):
        x = F.normalize(feature,dim=1)
        w = F.normalize(self.W,dim=0)
        cos = torch.matmul(x,w)/10
        a=  torch.acos(cos)
        top = torch.exp(s*torch.cos(a+m))
        down = torch.sum(torch.exp(s*torch.cos(a)),dim=1,
                         keepdim=True)-torch.exp(s*torch.cos(a))
        out = torch.log(top/(top+down))
        return out
if __name__ == '__main__':
    arc = Arc(1000,3)
    data = torch.randn(4,1000)
    out = arc(data)
    print(out)
