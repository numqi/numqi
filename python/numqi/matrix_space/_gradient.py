try:
    from ._gradient_model import (DetectRankModel, DetectRankOneModel,
                            DetectOrthogonalRankOneModel, DetectOrthogonalRankOneEigenModel)
except ImportError:
    DetectRankModel = None
    # the following model is not public-api, use it at your own risk
    DetectRankOneModel = None
    DetectOrthogonalRankOneModel = None
    DetectOrthogonalRankOneEigenModel = None
