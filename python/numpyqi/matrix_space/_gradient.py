try:
    import torch
    from ._gradient_torch_only import DetectMatrixSpaceRank, DetectExtendibleChannel
except ImportError:
    torch = None
    DetectMatrixSpaceRank = None
    DetectExtendibleChannel = None
