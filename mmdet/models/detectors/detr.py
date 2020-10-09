from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class DETR(SingleStageDetector):

    def __init__(self,
                 backbone,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DETR, self).__init__(backbone, None, bbox_head, train_cfg,
                                   test_cfg, pretrained)
        # TODO check extract_feat
