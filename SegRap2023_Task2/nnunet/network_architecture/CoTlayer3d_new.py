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

class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(CotLayer, self).__init__()
        
        # dim为通道数, kernel_size可设为3
        # key特征的提取
        self.dim = dim
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(
            nn.Conv3d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=4, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True)
        )

        # qk拼接后的两组卷积W_theta，W_sigma，前者有ReLU，后者无激活层
        # 两个1×1卷积层：CBL + CG
        # 1. 第二个卷积层没有ReLU，可以避免屏蔽过多的有效信息
        # 2. 第二个卷积层使用GroupNorm，将channel分为G组，每组分别做归一化，避免不同batch特征之间造成干扰
        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv3d(2*dim, dim//factor, 1, bias=False),
            nn.BatchNorm3d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim//factor, dim, kernel_size=1),#pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=dim)#pow(kernel_size, 2) * dim // share_planes)
        )

        # K2=A*v，对齐A和v的通道
        # self.convChannel = nn.Conv3d(pow(kernel_size, 2) * dim // share_planes, dim, 1)

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(dim)
        )

        # self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm3d(dim)
        
        # act = get_act_layer('swish')
        self.act = Swish(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32) # max(128*2//4, 32)=64
        self.se_manybatch = nn.Sequential(
            nn.Conv3d(dim, attn_chs, 1),
            nn.BatchNorm3d(attn_chs), #因为1C111会报错，1C112不会，相当于只有一个batch一个元素，没办法Norm
            nn.ReLU(inplace=True),
            nn.Conv3d(attn_chs, self.radix*dim, 1)
        )
        self.se_siglebatch = nn.Sequential(
            nn.Conv3d(dim, attn_chs, 1),
            # nn.BatchNorm3d(attn_chs), #因为1C111会报错，1C112不会，相当于只有一个batch一个元素，没办法Norm
            nn.ReLU(inplace=True),
            nn.Conv3d(attn_chs, self.radix*dim, 1)
        )

    def forward(self, x):
        # 1.1key map
        k = self.key_embed(x) #[2,128,50,100,100]

        # 1.2concat q(=x) and k
        qk = torch.cat([x, k], dim=1) #[2,256,50,100,100]
        b, c, qk_dd, qk_hh, qk_ww = qk.size()

        # 1.3经过两个1*1卷积，并展开为多头部
        A = self.embed(qk) #[2,128*9//8=144,50,100,100]
        ## A = A.view(b, 1, -1, self.kernel_size*self.kernel_size, qk_dd, qk_hh, qk_ww) #[2,1,16,9,50,100,100]

        # 2.1 generate v
        # conv1*1 + BN
        v = self.conv1x1(x) #[2,128,50,100,100]

        # 2.2 w和v进行处理并激活
        ## K2 = self.local_conv(v, A)
        # convChannel = nn.Conv3d(A.shape[1], k.shape[1], 1)
        # A = self.convChannel(A)
        K2 = v*A
        # K2 = v
        K2 = self.bn(K2)
        K2 = self.act(K2)

        # 3.1 k,wv 拼接处理
        B, C, D, H, W = K2.shape
        K2 = K2.view(B, C, 1, D, H, W) #[2,128,1,50,100,100]
        k = k.view(B, C, 1, D, H, W) #[2,128,1,50,100,100]
        Y = torch.cat([K2, k], dim=2) #[2,128,2,50,100,100]

        x_gap = Y.sum(dim=2) #[2,128,50,100,100] 等价于x+q(view之前的x和q)
        x_gap = x_gap.mean((2, 3, 4), keepdim=True) #[2,128,1,1,1]
        
        # 3.2 得到x_attn权重矩阵，Fusion静态和动态特征
        # conv1*1->BN->ReLU->conv1*1: dim->2*dim
        if int(x_gap.size(0)) < 2:
          x_attn = self.se_siglebatch(x_gap) #[2,256,1,1,1]
        else:
          x_attn = self.se_manybatch(x_gap)
        x_attn = x_attn.view(B, C, self.radix) #[2,128,2]
        x_attn = F.softmax(x_attn, dim=2) #[2,128,2]
        out = (Y * x_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)
        out += x
        
        return out.contiguous()


if __name__ == "__main__":
    x = torch.randn(1,128,50,100,100)
    cot = CotLayer(128,3)
    print(cot(x).shape)