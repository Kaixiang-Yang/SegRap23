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

class CoT(nn.Module):
    # Contextual Transformer Networks https://arxiv.org/abs/2107.12292
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)
        
        return k1+k2

class CotLayer(nn.Module):
    # Contextual Transformer Networks https://arxiv.org/abs/2107.12292
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv3d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv3d(dim,dim,1,bias=False),
            nn.BatchNorm3d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv3d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm3d(2*dim//factor),
            nn.ReLU(),
            nn.Conv3d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,d,h,w=x.shape
        k1=self.key_embed(x) #bs,c,d,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,d*h*w

        y=torch.cat([k1,x],dim=1) #bs,2c,d,h,w
        att=self.attention_embed(y) #bs,c*k*k,d,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,d,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,d*h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,d,h,w)
        
        return k1+k2+x
