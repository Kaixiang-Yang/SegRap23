import torch
import torch.nn.functional as F
from torch import nn

class DenseASPPBlock(nn.Sequential):
    """Conv Net block for building DenseASPP"""

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True, norm='in'):
        super(DenseASPPBlock, self).__init__()
        if bn_start:
            if norm == 'bn':
                self.add_module('norm_1', nn.BatchNorm3d(input_num))
            elif norm == 'in':
                self.add_module('norm_1', nn.InstanceNorm3d(input_num))
            
        self.add_module('relu_1', nn.LeakyReLU(negative_slope=0.01, inplace=True))
        self.add_module('conv_1', nn.Conv3d(in_channels=input_num, out_channels=num1, kernel_size=1))

        if norm == 'bn':
            self.add_module('norm_2', nn.BatchNorm3d(num1))
        elif norm == 'in':
            self.add_module('norm_2', nn.InstanceNorm3d(num1))
        self.add_module('relu_2', nn.LeakyReLU(negative_slope=0.01, inplace=True))
        self.add_module('conv_2', nn.Conv3d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate))

        self.drop_rate = drop_out

    def forward(self, input):
        feature = super(DenseASPPBlock, self).forward(input)

        if self.drop_rate > 0:
            feature = F.dropout3d(feature, p=self.drop_rate, training=self.training)

        return feature
    
class DenseASPP(nn.Module):
    def __init__(self, current_numfeature=64, feature0=64, feature1=32, norm='in'):
        super().__init__()
        d_feature0 = max(feature0, current_numfeature // 2)
        d_feature1 = max(feature1, current_numfeature // 4)
        dropout0 = 0

        self.ASPP_1 = DenseASPPBlock(input_num=current_numfeature, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(1,3,3), drop_out=dropout0, norm=norm)
        
        self.ASPP_2 = DenseASPPBlock(input_num=current_numfeature+d_feature1*1, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(1,6,6), drop_out=dropout0, norm=norm)
        
        self.ASPP_3 = DenseASPPBlock(input_num=current_numfeature+d_feature1*2, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(1,12,12), drop_out=dropout0, norm=norm)
        
        self.ASPP_4 = DenseASPPBlock(input_num=current_numfeature+d_feature1*3, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(1,18,18), drop_out=dropout0, norm=norm)
        
        self.conv = nn.Conv3d(current_numfeature+d_feature1*4, current_numfeature, 3, 1, 1)
        self.instance = nn.InstanceNorm3d(current_numfeature)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        aspp1 = self.ASPP_1(x)
        feature = torch.cat((aspp1, x), dim=1)

        aspp2 = self.ASPP_2(feature)
        feature = torch.cat((aspp2, feature), dim=1)

        aspp3 = self.ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp4 = self.ASPP_4(feature)
        feature = torch.cat((aspp4, feature), dim=1)

        out = self.lrelu(self.instance(self.conv(feature)))

        return out

    
if __name__ == "__main__":
    x = torch.randn(2,128,12,12,12).cuda()
    net = DenseASPP(128).cuda()
    print(net(x).shape)