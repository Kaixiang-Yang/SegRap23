#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, se=False, norm='bn'):
        super(inconv, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = SEBasicBlock(out_ch, out_ch, kernel_size=(1,3,3), norm=norm)

    def forward(self, x): 

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        return out 

class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Sequential(
                nn.Conv3d(channel, channel//reduction, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel, kernel_size=1, stride=1),
                nn.Sigmoid()
                )
    def forward(self, x):

        y = self.avg_pool(x)
        y = self.conv(y)

        return x * y

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation_rate=1):
    if kernel_size == (1,3,3):
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, \
                padding=(0,1,1), bias=False, dilation=dilation_rate)

    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,\
                padding=padding, bias=False, dilation=dilation_rate)
    
class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, reduction=4, dilation_rate=1, norm='bn'):
        super(SEBasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, kernel_size=kernel_size, stride=stride)
        if norm == 'bn':
            self.bn1 = nn.BatchNorm3d(inplanes)
        elif norm =='in':
            self.bn1 = nn.InstanceNorm3d(inplanes)
        else:
            raise ValueError('unsupport norm method')
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=dilation_rate)
        if norm == 'bn':
            self.bn2 = nn.BatchNorm3d(planes)
        elif norm =='in':
            self.bn2 = nn.InstanceNorm3d(planes)
        else:
            raise ValueError('unsupport norm method')
        self.se = SELayer(planes, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            if norm == 'bn':
                self.shortcut = nn.Sequential(
                    nn.BatchNorm3d(inplanes),
                    self.relu,
                    nn.Conv3d(inplanes, planes, kernel_size=1, \
                            stride=stride, bias=False)
                )
            elif norm =='in':
                self.shortcut = nn.Sequential(
                    nn.InstanceNorm3d(inplanes),
                    self.relu,
                    nn.Conv3d(inplanes, planes, kernel_size=1, \
                            stride=stride, bias=False)
                )
            else:
                raise ValueError('unsupport norm method')

        self.stride = stride

    def forward(self, x):
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.se(out)

        out += self.shortcut(residue)

        return out
    
class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, se=False, reduction=2, dilation_rate=1, norm='bn'):
        super(conv_block, self).__init__()

        self.conv = SEBasicBlock(in_ch, out_ch, stride=stride, reduction=reduction, dilation_rate=dilation_rate, norm=norm)

    def forward(self, x):

        out = self.conv(x)

        return out

class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, scale=(2,2,2), se=False, reduction=2, norm='bn'):
        super(up_block, self).__init__()

        self.scale = scale

        self.conv = nn.Sequential(
            conv_block(in_ch+out_ch, out_ch, se=se, reduction=reduction, norm=norm),
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=self.scale, mode='trilinear', align_corners=True)

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)

        return out

class DenseASPPBlock(nn.Sequential):
    """Conv Net block for building DenseASPP"""

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True, norm='bn'):
        super(DenseASPPBlock, self).__init__()
        if bn_start:
            if norm == 'bn':
                self.add_module('norm_1', nn.BatchNorm3d(input_num))
            elif norm == 'in':
                self.add_module('norm_1', nn.InstanceNorm3d(input_num))
            
        self.add_module('relu_1', nn.ReLU(inplace=True))
        self.add_module('conv_1', nn.Conv3d(in_channels=input_num, out_channels=num1, kernel_size=1))

        if norm == 'bn':
            self.add_module('norm_2', nn.BatchNorm3d(num1))
        elif norm == 'in':
            self.add_module('norm_2', nn.InstanceNorm3d(num1))
        self.add_module('relu_2', nn.ReLU(inplace=True))
        self.add_module('conv_2', nn.Conv3d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate))

        self.drop_rate = drop_out

    def forward(self, input):
        feature = super(DenseASPPBlock, self).forward(input)

        if self.drop_rate > 0:
            feature = F.dropout3d(feature, p=self.drop_rate, training=self.training)

        return feature
    
class Generic_UNet_ASPP(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=None,
                 seg_output_use_bias=False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_UNet_ASPP, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        se = True
        reduction = 2
        norm = 'bn'

        # downsample twice
        self.share_conv1x = inconv(input_channels, 24, norm=norm)

        self.share_conv1x_2 = self._make_layer(
            conv_block,  24, 32, 2, se=se, stride=1, reduction=reduction, norm=norm)
        self.share_maxpool1 = nn.MaxPool3d((1, 2, 2))

        self.share_conv2x = self._make_layer(
            conv_block, 32, 48, 2, se=se, stride=1, reduction=reduction, norm=norm)
        self.share_maxpool2 = nn.MaxPool3d((2, 2, 2)) 

        self.share_conv4x = self._make_layer(
            conv_block, 48, 64, 2, se=se, stride=1, reduction=reduction, norm=norm)

        # DenseASPP
        current_num_feature = 64
        d_feature0 = 64
        d_feature1 = 32
        dropout0 = 0 
        self.share_ASPP_1 = DenseASPPBlock(input_num=current_num_feature, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(1, 3, 3), drop_out=dropout0, norm=norm)

        self.share_ASPP_2 = DenseASPPBlock(input_num=current_num_feature+d_feature1*1, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(1, 5, 5), drop_out=dropout0, norm=norm)

        self.share_ASPP_3 = DenseASPPBlock(input_num=current_num_feature+d_feature1*2, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(1, 7, 7), drop_out=dropout0, norm=norm)

        self.share_ASPP_4 = DenseASPPBlock(input_num=current_num_feature+d_feature1*3, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(1, 9, 9), drop_out=dropout0, norm=norm)
        current_num_feature = current_num_feature + 4 * d_feature1

        # upsample
        self.share_up1 = up_block(in_ch=current_num_feature,
                               out_ch=48, se=se, reduction=reduction, norm=norm)
        self.share_literal1 = nn.Conv3d(48, 48, 3, padding=1)

        self.share_up2 = up_block(in_ch=48, out_ch=32, scale=(
            1, 2, 2), se=se, reduction=reduction, norm=norm)
        self.share_literal2 = nn.Conv3d(32, 32, 3, padding=1)
        # branch
        self.out_conv = nn.Conv3d(32, self.num_classes, 1, 1)

    def forward(self, x):
        # down
        x1 = self.share_conv1x(x[:, -1:, :, :, :])

        o1 = self.share_conv1x_2(x1)

        o2 = self.share_maxpool1(o1)
        o2 = self.share_conv2x(o2)
        o3 = self.share_maxpool2(o2)
        o3 = self.share_conv4x(o3)

        # DenseASPP
        aspp1 = self.share_ASPP_1(o3)
        feature = torch.cat((aspp1, o3), dim=1)

        aspp2 = self.share_ASPP_2(feature)
        feature = torch.cat((aspp2, feature), dim=1)

        aspp3 = self.share_ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp4 = self.share_ASPP_4(feature)
        feature = torch.cat((aspp4, feature), dim=1)

        out = self.share_up1(feature, self.share_literal1(o2))
        out = self.share_up2(out, self.share_literal2(o1))

        out = self.out_conv(out)

        if self._deep_supervision and self.do_ds:
            return tuple([out])
        else:
            return out

    def _make_layer(self, block, in_ch, out_ch, num_blocks, se=True, stride=1, reduction=2, dilation_rate=1, norm='bn'):
        layers = []
        layers.append(block(in_ch, out_ch, se=se, stride=stride,
                            reduction=reduction, dilation_rate=dilation_rate, norm=norm))
        for i in range(num_blocks-1):
            layers.append(block(out_ch, out_ch, se=se, stride=1,
                                reduction=reduction, dilation_rate=dilation_rate, norm=norm))

        return nn.Sequential(*layers)
    
    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp
