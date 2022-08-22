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
from .utils import nested_tensor_from_tensor_list, NestedTensor, inverse_sigmoid,reset_parameters

class RelationFormer(nn.Module):
    """ This is the RelationFormer module that performs object detection """

    def __init__(self, backbone, transformer, config, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.config = config

        self.num_queries = config.MODEL.DECODER.NUM_QUERIES
        self.hidden_dim = config.MODEL.DECODER.HIDDEN_DIM

        self.num_feature_levels = config.MODEL.DECODER.NUM_FEATURE_LEVELS
        self.two_stage = config.MODEL.DECODER.TWO_STAGE
        self.aux_loss = config.MODEL.DECODER.AUX_LOSS
        self.with_box_refine = config.MODEL.DECODER.WITH_BOX_REFINE
        self.focal_loss = not config.TRAIN.FOCAL_LOSS_ALPHA==''
        self.num_classes = config.MODEL.NUM_OBJ_CLS if self.focal_loss else config.MODEL.NUM_OBJ_CLS+1

        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 2*config.DATA.DIM, 3)

        self.add_emd_rel = config.MODEL.DECODER.ADD_EMB_REL
        self.use_dropout = config.MODEL.DECODER.DROPOUT_REL
        if self.use_dropout: # this dropout sub be used only for obj and freq bias to force rel token to learn
            self.dropout_rel = nn.Dropout(p=config.MODEL.DECODER.DROPOUT)
        #relation embedding
        num_of_token = 3
        input_dim = (self.hidden_dim*num_of_token +8 +2*self.num_classes) if self.add_emd_rel else self.hidden_dim*num_of_token #+2*self.num_classes
        feed_fwd = config.MODEL.DECODER.DIM_FEEDFORWARD if hasattr(config.MODEL.DECODER, 'NORM_REL_EMB') and config.MODEL.DECODER.NORM_REL_EMB else self.hidden_dim
        self.relation_embed = MLP(input_dim, feed_fwd, config.MODEL.NUM_REL_CLS+1, 3,use_norm=hasattr(config.MODEL.DECODER, 'NORM_REL_EMB') and config.MODEL.DECODER.NORM_REL_EMB,
        dropout=config.MODEL.DECODER.DROPOUT)
        if not self.two_stage:
            self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim*2)    # why *2
        if self.num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_]
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
                    nn.Conv2d(self.backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])
        if config.TRAIN.FOCAL_LOSS_ALPHA:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)

        num_pred = (self.transformer.decoder.num_layers + 1) if self.two_stage else self.transformer.decoder.num_layers
        
        if self.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            # self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            # self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if self.two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)


    def forward(self, samples):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        #samples = nested_tensor_from_tensor_list([tensor.expand(3, -1, -1).contiguous() for tensor in samples])

        # Deformable Transformer backbone
        features, pos = self.backbone(samples)
        
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
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
    
        hs, init_reference, inter_references, attn_map, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            srcs, masks, query_embeds, pos
        )
        object_token = hs[..., :-1, :]
        if self.with_box_refine:
            outputs_classes = []
            outputs_coords = []
            for lvl in range(object_token.shape[0]):
                if lvl == 0:
                    reference = init_reference[...,:-1,:]
                else:
                    reference = inter_references[lvl - 1][...,:-1,:]
                reference = inverse_sigmoid(reference)
                outputs_class = self.class_embed[lvl](object_token[lvl])
                tmp = self.bbox_embed[lvl](object_token[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            outputs_class = torch.stack(outputs_classes)
            outputs_coord = torch.stack(outputs_coords)
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        else:
            outputs_class = self.class_embed(object_token)
            outputs_coord = self.bbox_embed(object_token).sigmoid()
        
            out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        cls_emb = self.dropout_rel(hs[...,:-1,:]) if self.use_dropout else hs[...,:-1,:]
        rel_emb = hs[...,-1,:]
        out['attn_map'] = attn_map

        if self.add_emd_rel:
            cls_emb = torch.cat((cls_emb,outputs_class,outputs_coord),-1)#

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        return (cls_emb[-1],rel_emb[-1]) if not (self.training and self.aux_loss) else (cls_emb,rel_emb), out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, use_norm=False, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.use_norm = False
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        if use_norm:
            self.use_norm = True
            self.layers.insert(0,nn.LayerNorm(input_dim))
            # self.layers.insert(2,nn.Dropout(p=dropout))
            # self.layers.insert(4, nn.Dropout(p=dropout))
            self.dropout1 = nn.Dropout(p=dropout)
            self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, retun_interm=False):
        if self.use_norm:
            x = self.layers[3](self.dropout2(F.relu(self.layers[2](self.dropout1(self.layers[1](self.layers[0](x)))))))
        else:
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
                # if  retun_interm and i==self.num_layers-2 :
                #     interm_feats = x
            # if retun_interm:
            #     return x,interm_feats
            # else:
            #
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_relationformer(config, **kwargs):
    if 'swin' in config.MODEL.ENCODER.BACKBONE:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(config.MODEL.ENCODER.BACKBONE)
    else:
        backbone = build_backbone(config)
    transformer = build_deforamble_transformer(config)

    model = RelationFormer(
        backbone,
        transformer,
        config,
        **kwargs
    )

    return model