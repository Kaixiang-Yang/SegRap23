import math
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from torch.nn.modules.utils import _triple

import torch
from string import Template
from collections import namedtuple
import cupy

Stream = namedtuple('Stream', ['ptr'])

def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'

@cupy.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)

CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''

def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS

_aggregation_zeropad_forward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_zeropad_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${weight_heads} / ${input_channels} / ${top_height} / ${top_width};
    const int head = (index / ${top_width} / ${top_height} / ${input_channels}) % ${weight_heads};
    const int c = (index / ${top_width} / ${top_height}) % ${input_channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};

    ${Dtype} value = 0;
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
          const int offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
          const int offset_weight = (((n * ${weight_heads} + head) * ${weight_channels} + c % ${weight_channels}) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
          value += weight_data[offset_weight] * bottom_data[offset_bottom];
        }
      }
    }
    top_data[index] = value;
  }
}
'''

_aggregation_zeropad_input_backward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_zeropad_input_backward_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, ${Dtype}* bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${input_channels} / ${bottom_height} / ${bottom_width};
    const int c = (index / ${bottom_height} / ${bottom_width}) % ${input_channels};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = index % ${bottom_width};
    ${Dtype} value = 0;

    for (int head = 0; head < ${weight_heads}; ++head) {
        for (int kh = 0; kh < ${kernel_h}; ++kh) {
          for (int kw = 0; kw < ${kernel_w}; ++kw) {
            const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
            const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
            if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
              const int h_out = h_out_s / ${stride_h};
              const int w_out = w_out_s / ${stride_w};
              if ((h_out >= 0) && (h_out < ${top_height}) && (w_out >= 0) && (w_out < ${top_width})) {
                const int offset_top = (((n * ${weight_heads} + head) * ${input_channels} + c) * ${top_height} + h_out) * ${top_width} + w_out;
                const int offset_weight = (((n * ${weight_heads} + head) * ${weight_channels} + c % ${weight_channels}) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
                value += weight_data[offset_weight] * top_diff[offset_top];
              }
            }
          }
        }
    }
    bottom_diff[index] = value;
  }
}
'''

_aggregation_zeropad_weight_backward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_zeropad_weight_backward_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* weight_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${weight_heads} / ${weight_channels} / ${top_height} / ${top_width};
    const int head = (index / ${top_width} / ${top_height} / ${weight_channels}) % ${weight_heads};
    const int c = (index / ${top_width} / ${top_height}) % ${weight_channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};

    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        const int offset_weight = (((n * ${weight_heads} + head) * ${weight_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
        ${Dtype} value = 0;
        if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
          for (int cc = c; cc < ${input_channels}; cc += ${weight_channels}) {
            const int offset_bottom = ((n * ${input_channels} + cc) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
            const int offset_top = (((n * ${weight_heads} + head) * ${input_channels} + cc) * ${top_height} + h) * ${top_width} + w;
            value += bottom_data[offset_bottom] * top_diff[offset_top];
          }
        }
        weight_diff[offset_weight] = value;
      }
    }
  }
}
'''

class AggregationZeropad(Function):
    @staticmethod
    def forward(ctx, input, weight, kernel_size, stride, padding, dilation):
        kernel_size, stride, padding, dilation = _triple(kernel_size), _triple(stride), _triple(padding), _triple(dilation)
        ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation = kernel_size, stride, padding, dilation
        assert input.dim() == 5 and input.is_cuda and weight.is_cuda
        batch_size, input_channels, input_depth, input_height, input_width = input.size()
        _, weight_heads, weight_channels, weight_kernels, weight_depth, weight_height, weight_width = weight.size()
        output_depth = int((input_depth + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
        output_height = int((input_height + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
        output_width = int((input_width + 2 * padding[2] - (dilation[2] * (kernel_size[2] - 1) + 1)) / stride[2] + 1)
        assert output_depth * output_height * output_width == weight_depth * weight_height * weight_width
        output = input.new(batch_size, weight_heads * input_channels, output_depth, output_height, output_width)
        n = output.numel()
        if not input.is_contiguous():
            input = input.detach().clone()
        if not weight.is_contiguous():
            weight = weight.detach().clone()

        with torch.cuda.device_of(input):
            f = load_kernel('aggregation_zeropad_forward_kernel', _aggregation_zeropad_forward_kernel, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, input_channels=input_channels, 
                            weight_heads=weight_heads, weight_channels=weight_channels,
                            bottom_depth=input_depth, bottom_height=input_height, bottom_width=input_width,
                            top_depth=output_depth, top_height=output_height, top_width=output_width,
                            kernel_d=kernel_size[0], kernel_h=kernel_size[1], kernel_w=kernel_size[2],
                            stride_d=stride[0], stride_h=stride[1], stride_w=stride[2],
                            dilation_d=dilation[0], dilation_h=dilation[1], dilation_w=dilation[2],
                            pad_d=padding[0], pad_h=padding[1], pad_w=padding[2])
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel_size, stride, padding, dilation = ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation
        input, weight = ctx.saved_tensors
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        batch_size, input_channels, input_depth, input_height, input_width = input.size()
        _, weight_heads, weight_channels, weight_kernels, weight_depth, weight_height, weight_width = weight.size()
        output_depth, output_height, output_width = grad_output.size()[2:]
        grad_input, grad_weight = None, None
        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, input_channels=input_channels, 
                   weight_heads=weight_heads, weight_channels=weight_channels,
                   bottom_depth=input_depth, bottom_height=input_height, bottom_width=input_width,
                   top_depth=output_depth, top_height=output_height, top_width=output_width,
                   kernel_d=kernel_size[0], kernel_h=kernel_size[1], kernel_w=kernel_size[2],
                   stride_d=stride[0], stride_h=stride[1], stride_w=stride[2],
                   dilation_d=dilation[0], dilation_h=dilation[1], dilation_w=dilation[2],
                   pad_d=padding[0], pad_h=padding[1], pad_w=padding[2])
        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())
                n = grad_input.numel()
                opt['nthreads'] = n
                f = load_kernel('aggregation_zeropad_input_backward_kernel', _aggregation_zeropad_input_backward_kernel, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
            if ctx.needs_input_grad[1]:
                grad_weight = weight.new(weight.size())
                n = grad_weight.numel() // weight.shape[3]
                opt['nthreads'] = n
                f = load_kernel('aggregation_zeropad_weight_backward_kernel', _aggregation_zeropad_weight_backward_kernel, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return grad_input, grad_weight, None, None, None, None

def aggregation_zeropad(input, weight, kernel_size=3, stride=1, padding=0, dilation=1):
    assert input.shape[0] == weight.shape[0] and (input.shape[1] % weight.shape[2] == 0)
    if input.is_cuda:
        out = AggregationZeropad.apply(input, weight, kernel_size, stride, padding, dilation)
    else:
        #raise NotImplementedError
        out = AggregationZeropad.apply(input.cuda(), weight.cuda(), kernel_size, stride, padding, dilation)
        torch.cuda.synchronize()
        out = out.cpu()
    return out

class LocalConvolution(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        pad_mode: int = 0,
    ):
        super(LocalConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_mode = pad_mode

    def forward(self, input: Tensor, weight: Tensor):
        #if self.pad_mode == 0:
        out = aggregation_zeropad(
            input, 
            weight, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation)
        #else:
        #  out = aggregation_refpad(
        #    input, 
        #    weight, 
        #    kernel_size=self.kernel_size, 
        #    stride=self.stride, 
        #    padding=self.padding, 
        #    dilation=self.dilation)  
        return out

def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid()) # x = x.*sigmoid(x)


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)

class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CotLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv3d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=4, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv3d(2*dim, dim//factor, 1, bias=False),
            nn.BatchNorm3d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim//factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)
        self.convchannel = nn.Conv3d(pow(kernel_size, 2) * dim // share_planes, dim, 3, padding=1)#adding self
        self.bn = nn.BatchNorm3d(dim)
        
        # act = get_act_layer('swish')
        self.act = Swish(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv3d(dim, attn_chs, 1),
            # nn.BatchNorm3d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv3d(attn_chs, self.radix*dim, 1)
        )

    def forward(self, x):
        k = self.key_embed(x)
        qk = torch.cat([x, k], dim=1)
        b, c, qk_dd, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        w = w.view(b, 1, -1, self.kernel_size*self.kernel_size, qk_dd, qk_hh, qk_ww)#ori
        
        x = self.conv1x1(x)
        # w = self.convchannel(w)#new适配通道
        # x = x*w#通道乘积
        x = self.local_conv(x, w)#ori
        x = self.bn(x)
        x = self.act(x)

        B, C, D, H, W = x.shape
        x = x.view(B, C, 1, D, H, W)
        k = k.view(B, C, 1, D, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3, 4), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)
        
        return out.contiguous()


if __name__ == "__main__":
    net = nn.Sequential(
      nn.Conv3d(1,16,3,1,1),
      CotLayer(16,3),
      # nn.Linear(-1,3)
      nn.Conv3d(16,10,3,1,1),
      nn.Softmax(dim=1)
    ).cuda()
    x = torch.randn(4,1,64,64,64).cuda()
    y = torch.randn(4,10,64,64,64).cuda()
    mse = nn.MSELoss(reduction='mean')
    out = net(x)
    loss = mse(out,y)
    # loss.backward()
    print(loss)