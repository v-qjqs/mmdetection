import torch

from mmdet.core import bbox2result
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

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        assert len(x) == 1
        x = x[-1]
        pad_h, pad_w, _ = img_metas[0]['pad_shape']
        img_h, img_w, _ = img_metas[0]['img_shape']
        # num_imgs = len(img_metas)
        masks = torch.ones((1, pad_h, pad_w)).to(x.device)
        # for i in range(num_imgs):
        masks[0, :img_h, :img_w] = 0
        masks = masks.to(x.dtype)
        outs = self.bbox_head(x, masks)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results
