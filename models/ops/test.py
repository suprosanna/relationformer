# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 . All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/fundamentalvision/Deformable-DETR
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch


# Large
#N, M, C = 1, 36, 32
#Lq, L, P = 10800, 3 ,4
#shapes = torch.as_tensor([(16, 15, 39), (8, 8, 20), (4, 4, 10)], dtype=torch.long).cuda()



# Medium
#N, M, C = 1, 16, 16 # M6,26, C32
#Lq, L, P = 4860, 3 ,4
#shapes = torch.as_tensor([(8, 15, 39), (4, 4, 10), (2, 2, 5)], dtype=torch.long).cuda()


# Small
N, M, C = 2, 3, 4
Lq, L, P = 4, 2 ,4
shapes = torch.as_tensor([(3, 6, 4), (2, 3, 2)], dtype=torch.long).cuda()

# Tiny
#N, M, C = 1,1,1    #samples N, attention heads M, channels C
#Lq, L, P = 1,1,1   #query numbers/features Lq , Layers L (for multi-shape), reference ponits P
#shapes = torch.as_tensor([(2,2,2)], dtype=torch.long).cuda() # DxHxW shapes. Why DHW not HWD: https://discuss.pytorch.org/t/why-use-dxhxw-for-3d-input-data-instead-of-hxwxd/104045

level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1])) #start indices of input values in a linear array of all pixels
S = sum([(D*H*W).item() for D, H, W in shapes]) # total voxel count S


#torch.manual_seed(5)

#profiling computational speed of forward model.
@torch.no_grad()
def profile_forward_double(iters = 100, cuda=True):
    #data is just generated once for all interations as we neglect as most data transmissions as possible and only focus on computational speed here.
    value = torch.rand(N, S, M, C).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 3).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True) #normalize
    im2col_step = 2

    profiler.start()
    for i in range(iters):
        if(cuda):
            output_cuda = MSDeformAttnFunction.apply(value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step).detach().cpu()
        else:
            output_pytorch = ms_deform_attn_core_pytorch(value.double(), shapes, sampling_locations.double(), attention_weights.double()).detach().cpu()
    profiler.stop()

@torch.no_grad()
def check_forward_equal_with_pytorch_double():
    value = torch.rand(N, S, M, C).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 3).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True) #normalize
    im2col_step = 2
    output_pytorch = ms_deform_attn_core_pytorch(value.double(), shapes, sampling_locations.double(), attention_weights.double()).detach().cpu()
    output_cuda = MSDeformAttnFunction.apply(value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step).detach().cpu()
    fwdok = torch.allclose(output_cuda, output_pytorch)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    print(f'* {fwdok} check_forward_equal_with_pytorch_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


@torch.no_grad()
def check_forward_equal_with_pytorch_float():
    value = torch.rand(N, S, M, C).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 3).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    output_pytorch = ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights).detach().cpu()
    output_cuda = MSDeformAttnFunction.apply(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step).detach().cpu()
    fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    print(f'* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


def check_gradient_numerical(channels=4, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True):

    value = torch.rand(N, S, M, channels).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 3).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    func = MSDeformAttnFunction.apply

    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight

    gradok = gradcheck(func, (value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step))

    print(f'* {gradok} check_gradient_numerical(C={channels})')


if __name__ == '__main__':
    check_forward_equal_with_pytorch_double()
    check_forward_equal_with_pytorch_float()

    for channels in [1,2,3,4,5,6,7,8,9,10,32,64,65,66,67,68,69,70,71,128,256,1024,1025,2048,2049]:
        check_gradient_numerical(channels, True, True, True)

    # Use "nvprof -o forward_cuda.nvvp -f --profile-from-start off python3 test.py" with nvvp
    import torch.cuda.profiler as profiler
    profile_forward_double(100,True)    #True = cuda version, False pytorch version
