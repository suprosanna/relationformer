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

from .detr_transformer_3D import build_detr_transformer
from .seresnet import build_seresnet
from .swin_transformer_3D import build_swin_transformer
from .position_encoding import PositionEmbeddingSine3D, PositionEmbeddingLearned

class RelationFormer(nn.Module):
    """ This is the RelationFormer module that performs object detection """

    def __init__(self, encoder, decoder, config, cls_token=False, use_proj_in_dec=False, device='cuda'):
        """[summary]

        Args:
            encoder ([type]): [description]
            decoder ([type]): [description]
            num_classes (int, optional): [description]. Defaults to 8.
            num_queries (int, optional): [description]. Defaults to 100.
            imsize (int, optional): [description]. Defaults to 64.
            cls_token (bool, optional): [description]. Defaults to False.
            use_proj_in_dec (bool, optional): [description]. Defaults to False.
            with_box_refine (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = config.MODEL.DECODER.NUM_QUERIES
        # self.imsize = imsize
        self.cls_token = cls_token
        self.hidden_dim = config.MODEL.DECODER.HIDDEN_DIM
        self.device = device
        self.init = True
        self.use_proj_in_dec = use_proj_in_dec
        self.position_embedding = PositionEmbeddingSine3D(channels=config.MODEL.DECODER.HIDDEN_DIM)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        
        self.input_proj = nn.Conv3d(encoder.num_features, config.MODEL.DECODER.HIDDEN_DIM, kernel_size=1)
        
        self.class_embed = nn.Linear(config.MODEL.DECODER.HIDDEN_DIM, 2)
        self.coord_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM, config.MODEL.DECODER.HIDDEN_DIM, 6, 3)
        self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*3, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)


        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.class_embed.bias.data = torch.ones(2) * bias_value
        # nn.init.constant_(self.coord_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(self.coord_embed.layers[-1].bias.data, 0)


    def forward(self, samples):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        
        # swin transformer
        # feat_list, mask_list, pos_list = self.encoder(samples, self.position_embedding, return_interm_layers=False)
        
        # seresnet
        feat_list = [self.encoder(samples)]
        mask_list = [torch.zeros(feat_list[0][:, 0, ...].shape, dtype=torch.bool).to(feat_list[0].device)]
        pos_list = [self.position_embedding(mask_list[-1]).to(feat_list[0].device)]
        
        query_embed = self.query_embed.weight
        h = self.decoder(self.input_proj(feat_list[-1]), mask_list[-1], query_embed, pos_list[-1])
        object_token = h[...,:-1,:]

        class_prob = self.class_embed(object_token)
        coord_loc = self.coord_embed(object_token).sigmoid()
        
        out = {'pred_logits': class_prob, 'pred_nodes': coord_loc}
        return h, out


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


def build_relationformer(config, **kwargs):

    # encoder = build_swin_transformer(config)
    encoder = build_seresnet(config)

    decoder = build_detr_transformer(config)

    model = RelationFormer(
        encoder,
        decoder,
        config,
        **kwargs
    )

    # losses = ['labels', 'boxes', 'cardinality']

    # criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
    #                          eos_coef=args.eos_coef, losses=losses, loss_type = args.loss_type, use_fl=args.use_fl, loss_transform=args.loss_transform)
    # criterion.to(device)
    
    return model