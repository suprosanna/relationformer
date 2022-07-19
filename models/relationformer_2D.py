# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
RelationFormer model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
# from torchvision.ops import nms
import matplotlib.pyplot as plt
import math
import copy

from .deformable_detr_backbone import build_backbone
from .deformable_detr_2D import build_deforamble_transformer
from .utils import nested_tensor_from_tensor_list, NestedTensor, inverse_sigmoid

class RelationFormer(nn.Module):
    """ This is the RelationFormer module that performs object detection """

    def __init__(self, encoder, decoder, config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        self.num_queries = config.MODEL.DECODER.OBJ_TOKEN + config.MODEL.DECODER.RLN_TOKEN + config.MODEL.DECODER.DUMMY_TOKEN
        self.obj_token = config.MODEL.DECODER.OBJ_TOKEN
        self.hidden_dim = config.MODEL.DECODER.HIDDEN_DIM

        self.num_feature_levels = config.MODEL.DECODER.NUM_FEATURE_LEVELS
        self.two_stage = config.MODEL.DECODER.TWO_STAGE
        self.aux_loss = config.MODEL.DECODER.AUX_LOSS
        self.with_box_refine = config.MODEL.DECODER.WITH_BOX_REFINE
        self.num_classes = config.MODEL.NUM_CLASSES

        self.class_embed = nn.Linear(config.MODEL.DECODER.HIDDEN_DIM, 2)
        self.bbox_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM, config.MODEL.DECODER.HIDDEN_DIM, 4, 3)
        
        if config.MODEL.DECODER.RLN_TOKEN > 0:
            self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*3, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)
        else:
            self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*2, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)

        if not self.two_stage:
            self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim*2)    # why *2
        if self.num_feature_levels > 1:
            num_backbone_outs = len(self.encoder.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.encoder.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.encoder.num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])

        self.decoder.decoder.bbox_embed = None


    def forward(self, samples, seg=True):
        if not seg and not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        elif seg:
            samples = nested_tensor_from_tensor_list([tensor.expand(3, -1, -1).contiguous() for tensor in samples])
        # Deformable Transformer backbone
        features, pos = self.encoder(samples)
        
        # Create 
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.encoder[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
    
        hs, init_reference, inter_references, _, _ = self.decoder(
            srcs, masks, query_embeds, pos
        )

        object_token = hs[...,:self.obj_token,:]

        class_prob = self.class_embed(object_token)
        coord_loc = self.bbox_embed(object_token).sigmoid()
        
        out = {'pred_logits': class_prob, 'pred_nodes': coord_loc}
        return hs, out, srcs


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_relationformer(config, **kwargs):

    encoder = build_backbone(config)
    decoder = build_deforamble_transformer(config)

    model = RelationFormer(
        encoder,
        decoder,
        config,
        **kwargs
    )

    return model