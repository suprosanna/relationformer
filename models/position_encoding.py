# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
import math
import numpy as np
import torch
import torch.nn as nn

# from util.misc import NestedTensor


class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, channels=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.orig_channels = channels
        self.channels = int(np.ceil(channels/6)*2)
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        """[summary]

        Args:
            mask ([type]): [shape BxHxWxD]

        Raises:
            RuntimeError: [description]

        Returns:
            [type]: [description]
        """""""""
        not_mask = ~mask
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        z_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            x_embed = (x_embed - 0.5) / (x_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = (y_embed - 0.5) / (y_embed[:, :, -1:, :] + eps) * self.scale
            z_embed = (z_embed - 0.5) / (z_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.channels, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.channels)  # TODO: check 6 or something else?

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_z = z_embed[..., None] / dim_t
        
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(4)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(4)
        pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=4).flatten(4)
        pos = torch.cat((pos_y, pos_x, pos_z), dim=4).permute(0, 4, 1, 2, 3)
        return pos[:,:self.orig_channels,...]


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, channels=256):
        super().__init__()
        self.orig_channels = channels
        channels = int(np.ceil(channels/6)*2)
        self.row_embed = nn.Embedding(50, channels)
        self.col_embed = nn.Embedding(50, channels)
        self.dep_embed = nn.Embedding(50, channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        batch_size, _, h, w, d = tensor.shape
        i = torch.arange(h, device=tensor.device)
        j = torch.arange(w, device=tensor.device)
        k = torch.arange(d, device=tensor.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        z_emb = self.dep_embed(k)
        pos = torch.cat([
            x_emb.unsqueeze(1).unsqueeze(2).repeat(1, w, d, 1),
            y_emb.unsqueeze(0).unsqueeze(2).repeat(h, 1, d, 1),
            z_emb.unsqueeze(0).unsqueeze(1).repeat(h, w, 1, 1),
        ], dim=-1)[...,:self.orig_channels].permute(3, 0, 1, 2).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).flatten(2).flatten(2)
        return pos


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.orig_channels = channels
        channels = int(np.ceil(channels/6)*2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, ch, x, y, z)
        :return: Positional Encoding Matrix of size (batch_size, ch, x, y, z)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, _, x, y, z = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x,y,z,self.channels*3),device=tensor.device).type(tensor.type())
        emb[:,:,:,:self.channels] = emb_x
        emb[:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,2*self.channels:] = emb_z

        return emb[None,:,:,:,:self.orig_channels].repeat(batch_size, 1, 1, 1, 1).permute(0,4,1,2,3)


# def build_position_encoding(args):
#     N_steps = args.hidden_dim // 2
#     if args.position_embedding in ('v2', 'sine'):
#         # TODO find a better way of exposing other arguments
#         position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
#     elif args.position_embedding in ('v3', 'learned'):
#         position_embedding = PositionEmbeddingLearned(N_steps)
#     else:
#         raise ValueError(f"not supported {args.position_embedding}")

#     return position_embedding
