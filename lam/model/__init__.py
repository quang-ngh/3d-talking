"""
LAM Model Module

Contains the main LAM transformer model for generating 3D avatars from images.
Full LAM (encoder + transformer + GS3DRenderer) is in full_lam.FullLAMModel.
"""

from .lam import LAMModel, LAMConfig, DummyTransformer, DummyGaussianDecoder

__all__ = [
    "LAMModel",
    "LAMConfig",
    "DummyTransformer",
    "DummyGaussianDecoder",
]

try:
    from .transformer import TransformerDecoder
    __all__.append("TransformerDecoder")
except Exception:
    pass

try:
    from .full_lam import FullLAMModel
    __all__.append("FullLAMModel")
except Exception:
    FullLAMModel = None
