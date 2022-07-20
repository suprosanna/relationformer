# from .relationformer import build_relationformer
from .relationformer_2D import build_relationformer
from .detr_transformer_3D import build_detr_transformer


def build_model(config, **kwargs):
    return build_relationformer(config, **kwargs)