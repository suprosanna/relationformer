# from .relationformer import build_relationformer
from .relationformer_2D import build_relationformer


def build_model(config, **kwargs):
    return build_relationformer(config, **kwargs)