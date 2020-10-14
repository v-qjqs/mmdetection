import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer, xavier_init
from mmcv.runner import force_fp32

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply)
from mmdet.models.utils import FFN, build_position_encoding, build_transformer
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class TransformerHead(AnchorFreeHead):

    def __init__(
            self,
            num_classes,
            in_channels,
            num_fcs=2,
            transformer=dict(
                type='Transformer',
                embed_dims=256,
                num_heads=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                feedforward_channels=2048,
                dropout=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
                norm_cfg=dict(type='LN'),
                num_fcs=2,
                pre_norm=False,
                return_intermediate_dec=True),
            position_encoding=dict(
                type='SinePositionEmbedding', num_feats=128, normalize=True),
            loss_cls=dict(
                type='CrossEntropyLoss',
                # NOTE bg_cls_weight means relative classification
                # weight of the no-object class
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_giou=dict(type='GIoULoss', loss_weight=2.0),
            train_cfg=dict(
                assigner=dict(
                    type='HungarianMatcher',
                    cls_wei=1.,
                    bbox_wei=5.,
                    giou_wei=2.),
                pos_weight=-1),
            test_cfg=dict(max_per_img=100),
            **kwargs):
        super(AnchorFreeHead, self).__init__()
        use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        # NOTE sigmoid is not supported in DETR, since
        # background is needed for the matcher.
        assert not use_sigmoid_cls
        assert hasattr(transformer, 'embed_dims') and hasattr(
            position_encoding, 'num_feats')
        num_feats = position_encoding['num_feats']
        embed_dims = transformer['embed_dims']
        assert num_feats * 2 == embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {embed_dims}' \
            f' and {num_feats}.'
        assert test_cfg is not None and hasattr(test_cfg, 'max_per_img')
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None:
            assert isinstance(class_weight, float)
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float)
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if hasattr(loss_cls, 'bg_cls_weight'):
                loss_cls.pop('bg_cls_weight')
        if train_cfg:
            assert loss_cls['loss_weight'] == train_cfg.assigner['cls_wei'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == train_cfg.assigner['bbox_wei']
            assert loss_giou['loss_weight'] == train_cfg.assigner['giou_wei']
            self.assigner = build_assigner(train_cfg.assigner)
            # DETR sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes + 1
        self.in_channels = in_channels
        self.num_fcs = num_fcs
        self.transformer = transformer
        self.position_encoding = position_encoding
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_sigmoid_cls = use_sigmoid_cls
        self.embed_dims = embed_dims
        self.num_query = test_cfg['max_per_img']
        self.background_label = num_classes
        self.fp16_enabled = False
        # super(TransformerHead, self).__init__(
        #     num_classes,
        #     in_channels,
        #     stacked_convs=0,
        #     loss_cls=loss_cls,
        #     loss_bbox=loss_bbox,
        #     background_label=None,
        #     train_cfg=train_cfg,
        #     test_cfg=test_cfg,
        #     **kwargs)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_giou = build_loss(loss_giou)
        self._init_layers()

    def _init_layers(self):
        # NOTE the background class is not included in AnchorFreeHead,
        # so here add 1 to overwrite cls_out_channels
        # self.cls_out_channels += 1
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
        self.pos_enc = build_position_encoding(self.position_encoding)
        act_cfg = self.transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(act_cfg)
        self.transformer = build_transformer(self.transformer)

        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.reg_embed = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_fcs,
            act_cfg,
            add_residual=False)
        self.fc_reg = Linear(self.embed_dims, 4)
        self.query_embed = nn.Embedding(self.num_query, self.embed_dims)

    def init_weights(self, distribution='uniform'):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution=distribution)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # NOTE here use AnchorFreeHead instead of TransformerHead,
        # since AnchorFreeHead._load_from_state_dict should not be
        # called here. Invoking the default nn.Module._load_from_state_dict
        # is enough.
        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, feats, masks):
        return self.forward_single(feats, masks)

    def forward_single(self, x, masks):
        # x: [bs,c,h,w], mask: [bs,img_pad_h,img_pad_w]
        x = self.input_proj(x)
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        pos_embed = self.pos_enc(masks)
        outs_dec, _ = self.transformer(
            x, masks, self.query_embed.weight,
            pos_embed)  # [nb_dec, bs,num_query,embed_dim]
        all_cls_scores = self.fc_cls(
            outs_dec)  # [nb_dec, bs,num_query,nb_class]
        all_bbox_preds = self.fc_reg(self.activate(
            self.reg_embed(outs_dec))).sigmoid()  # [nb_dec, bs,num_query,4]
        num_dec_layers = all_cls_scores.size(0)
        all_cls_scores = [all_cls_scores[i] for i in range(num_dec_layers)]
        all_bbox_preds = [all_bbox_preds[i] for i in range(num_dec_layers)]
        all_cls_scores = [all_cls_scores[-1]] + all_cls_scores[:-1]
        all_bbox_preds = [all_bbox_preds[-1]] + all_bbox_preds[:-1]
        return all_cls_scores, all_bbox_preds

    @force_fp32(apply_to=('all_cls_scores', 'all_bbox_preds'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        # all_cls_scores: [num_dec_layer,bs,num_query,nb_class]
        # all_bbox_preds: [num_dec_layer,bs,num_query,4]
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        # TODO check detr use sum for loss_single here?
        losses_cls, losses_bbox, losses_giou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_giou=losses_giou)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        # labels: [bs, num_query], cls_scores: [bs, num_query, nb_class]
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = torch.Tensor([img_w, img_h, img_w,
                                   img_h]).unsqueeze(0).repeat(
                                       bbox_pred.size(0),
                                       1).to(bbox_pred.device)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        num_total_samples = num_total_neg + num_total_pos
        num_total_samples = max(num_total_samples, 1)
        # TODO num_total_pos all reduce like DETR official
        num_total_pos = max(num_total_pos, 1)
        # TODO check avg_factor in cls head.
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=num_total_samples)

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        # regression giou loss
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors
        loss_giou = self.loss_giou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_giou

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        # labels: [bs, num_query], label_weights, num_total_samples
        # bbox_targets: [bs, num_query, 4], bbox_weights
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_scores,
                           bbox_preds,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           gt_bboxes_ignore=None):
        num_bboxes = bbox_preds.size(0)
        # matcher and sampler
        assign_result = self.assigner.assign(bbox_preds, cls_scores, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels,
                                             img_meta)
        sampling_result = self.sampler.sample(assign_result, bbox_preds,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.background_label,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes, dtype=torch.float)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if self.train_cfg.pos_weight > 0:
            label_weights[pos_inds] = self.train_cfg.pos_weight

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_preds)
        bbox_weights = torch.zeros_like(bbox_preds)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']
        factor = torch.Tensor([img_w, img_h, img_w,
                               img_h]).unsqueeze(0).to(bbox_preds.device)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    # over-write because img_metas are needed as inputs to generates
    # masks, which are then used for the forward process.
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        assert proposal_cfg is None
        # only feature from last stage of the backbone is supported now.
        x = x[-1]
        # generates binary masks used for transformer
        pad_h, pad_w, _ = img_metas[0]['pad_shape']
        num_imgs = len(img_metas)
        masks = torch.ones((num_imgs, pad_h, pad_w)).to(x.device)
        for i in range(num_imgs):
            img_h, img_w, _ = img_metas[i]['img_shape']
            masks[i][:img_h, :img_w] = 0
        masks = masks.to(x.dtype)
        outs = self(x, masks)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    @force_fp32(apply_to=('all_cls_scores', 'all_bbox_preds'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   img_metas,
                   rescale=False):
        # use the output from the last decoder layer
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        assert len(cls_score) == len(bbox_pred)
        # exclude background
        scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        # TODO check clip shoud after rescale?
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)
        return det_bboxes, det_labels
