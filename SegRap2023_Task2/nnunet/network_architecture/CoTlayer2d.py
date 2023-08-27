import torch
from torch import nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """
    Swish - Described in: https://arxiv.org/abs/1710.05941
    SiLU(x) = x*sigmoid(x)
    """
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def swish(self, x, inplace: bool = False):
        return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid()) # x = x.*sigmoid(x)

    def forward(self, x):
        return self.swish(x, self.inplace)

class CotLayer2D(nn.Module):
    # Contextual Transformer Networks https://arxiv.org/abs/2107.12292
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )

        self.bn = nn.BatchNorm2d(dim)
        self.act = Swish(inplace=True)

        self.se = nn.Sequential(
            nn.Conv2d(dim, max(dim*2//4,32), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim*2//4,32), 2*dim, 1)
        )

    #diff: 1.relu inplace 2.att softmax
    def forward(self, x):
        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x)#bs,c,h,w .view(bs,c,-1) #bs,c,h*w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False)#bs,c,h,w .view(bs,c,-1) #bs,c,h*w
        k2 = att*v
        k2 = self.act(self.bn(k2))
        # k2=F.softmax(att,dim=-1)*v
        # k2=k2.view(bs,c,d,h,w)

        B,C,H,W = k2.shape
        k2 = k2.view(B,C,1,H,W)
        k1 = k1.view(B,C,1,H,W)
        Y = torch.cat([k2,k1], dim=2)#B,C,2,H,W

        x_gap = Y.sum(dim=2)
        x_gap = x_gap.mean((2,3), keepdim=True) #b,c,1,1
        x_attn = self.se(x_gap)#b,2*c,1,1
        x_attn = x_attn.view(B, C, 2)
        x_attn = F.softmax(x_attn, dim=2)
        out = (Y * x_attn.reshape(B,C,2,1,1)).sum(dim=2)
        out += x
        
        return out.contiguous()

        
if __name__ == "__main__":
    # net = CoT3D(64,3)
    net = CotLayer2D(64,3)
    x = torch.randn(1, 64, 6, 5)
    print(net(x).shape)