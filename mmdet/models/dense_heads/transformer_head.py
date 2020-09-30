import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear
from mmcv.runner import force_fp32

from mmdet.core import build_assigner, build_sampler, multi_apply
from mmdet.models.utils import FFN, build_position_encoding, build_transformer
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class TransformerHead(AnchorFreeHead):

    def __init__(
            self,
            num_classes,
            in_channels,
            embed_dims,
            num_query,
            num_fcs=3,
            aux_loss=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(
                type='L1loss',
                loss_weight=1.0),  # TODO check bbox_loss_coef in DETR
            loss_giou=dict(
                type='GIoULoss',
                loss_weight=1.0),  # TODO check giou_loss_coef in DETR
            **kwargs):
        super(TransformerHead, self).__init__(num_classes, in_channels,
                                              **kwargs)
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        assert not self.use_sigmoid_cls  # TODO
        assert self.background_label == num_classes  # TODO
        self.cls_out_channels = num_classes + 1
        self.sampling = False
        self.assigner = build_assigner(self.train_cfg.assigner)
        # DETR sampling=False so use PseudoSampler
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)
        # self.loss_cls = build_loss(loss_cls)
        # self.loss_bbox = build_loss(loss_bbox)
        self.loss_giou = build_loss(loss_giou)
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.aux_loss = aux_loss
        self.num_fcs = num_fcs

    def _init_layers(self, ):
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
        # self.pos_enc = PositionEmbeddingSine(num_pos_feats, normalize)
        self.pos_enc = build_position_encoding(
            self.train_cfg.position_encoding)
        self.transformer = build_transformer(self.train_cfg.transformer)
        self.cls_embed = Linear(self.embed_dims, self.cls_out_channels)
        # TODO FFN no residual version
        # self.reg_embed = build_ffn(self.train_cfg.ffn)
        # embed_dims, feedforward_channels, num_fcs=2,
        # act_cfg=dict(type='ReLU', inplace=True), dropout=0.0
        self.reg_embed = FFN(self.embed_dims, self.embed_dims, self.num_fcs)
        self.query_embed = nn.Embedding(self.num_query, self.embed_dims)

    def init_weights(self, ):
        pass

    def _load_from_state_dict(self):
        pass

    def forward(self, feats, mask):
        masks = [mask for _ in range(len(feats))]
        assert len(feats) == len(masks)
        return multi_apply(self.forward_single, feats, masks)

    def forward_single(self, x, mask):
        # x: [bs,c,h,w], mask: [bs,img_pad_h,img_pad_w]
        x = self.input_proj(x)
        mask = F.interpolate(
            mask.unsqueeze(1), size=x.shape[-2:]).to(torch.bool)
        pos_embed = self.pos_enc(mask)
        outs_dec, _ = self.transformer(x, mask, self.query_embed.weight,
                                       pos_embed)  # [bs,num_query,embed_dim]
        cls_score = self.cls_embed(outs_dec)  # [bs,num_query,nb_class]
        bbox_pred = self.reg_embed(outs_dec)  # [bs,num_query,4]
        return cls_score, bbox_pred

    @force_fp32(apply_to=('all_cls_scores', 'all_bbox_preds'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             gt_bboxes_list,
             gt_labels_list,
             gt_bboxes_ignore_list=None):
        # all_cls_scores: [num_dec_layer,bs,num_query,nb_class]
        # all_bbox_preds: [num_dec_layer,bs,num_query,4]
        num_dec_layers = all_cls_scores.size(0)
        all_cls_scores = [all_cls_scores[i] for i in range(num_dec_layers)]
        all_bbox_preds = [all_bbox_preds[i] for i in range(num_dec_layers)]
        all_cls_scores = [all_cls_scores[-1]] + all_cls_scores[:-1]
        all_bbox_preds = [all_bbox_preds[-1]] + all_bbox_preds[:-1]
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_dec_layers)
        ]
        losses_cls, losses_bbox, losses_giou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_bboxes_ignore_list)
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_giou=losses_giou)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        # TODO img_metas
        # labels: [bs, num_query], cls_scores: [bs, num_query, nb_class]
        assert gt_bboxes_ignore_list is None
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        num_total_samples = num_total_neg + num_total_pos
        num_total_samples = num_total_samples.clamp(min=1)
        # TODO num_total_pos all reduce like DETR official
        num_total_pos = num_total_pos.clamp(min=1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        # bbox_targets: [bs,num_query,4]
        bbox_preds = bbox_preds.reshape(-1, 4)
        # TODO check no bbox decoder.
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        # giou regression loss
        loss_giou = self.loss_giou(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        # TODO mask loss
        # return dict(loss_cls=loss_cls, loss_bbox=loss_bbox,
        # loss_giou=loss_giou)
        return loss_cls, loss_bbox, loss_giou

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        # labels: [bs, num_query], label_weights, num_total_samples
        # bbox_targets: [bs, num_query, 4], bbox_weights
        num_imgs = len(cls_scores_list)
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      bbox_preds_list, gt_bboxes_list,
                                      gt_labels_list, gt_bboxes_ignore_list)
        # num_total_pos = sum((max(inds.numel(), 1) for inds in pos_inds_list))
        # num_total_neg = sum((max(inds.numel(), 1) for inds in neg_inds_list))
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        # TODO check num_total_pos when no gt bbox. TODO check img_metas
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_scores,
                           bbox_preds,
                           gt_bboxes,
                           gt_labels,
                           gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None
        num_bboxes = bbox_preds.size(0)
        # num_total_samples = gt_bboxes.size(0)  # TODO img_meta?
        # matcher
        assign_result = self.assigner.assign(bbox_preds, cls_scores, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        # assigned_labels = assign_result.labels
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
        # assigned_gt_inds = assign_result.gt_inds
        # pos_inds = assigned_labels > -1
        # pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        # pos_inds = torch.nonzero(assigned_labels > -1)
        # labels[pos_inds] = assigned_labels[pos_inds]
        if self.train_cfg.pos_weight > 0:
            label_weights[pos_inds] = self.train_cfg.pos_weight
        # bbox targets
        bbox_targets = torch.zeros_like(bbox_preds)
        bbox_weights = torch.zeros_like(bbox_preds)
        bbox_weights[pos_inds] = 1.0
        # bbox_targets[pos_inds] = gt_bboxes[assign_result.gt_inds[pos_inds]-1]
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, ):
        pass
