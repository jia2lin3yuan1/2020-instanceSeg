import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from layers.modules.instance_loss import BinaryLoss, PermuInvLoss, PermuInvLossDilatedConv
from layers.modules.regularize_loss import QuantityLoss, MumfordShahLoss
from layers.modules.refine_loss import RefineLoss
from layers.modules.evaluate import Evaluate

class LossEvaluate(nn.Module):
    '''
    compute loss:
        1) binary loss to separate BG and FG
        2) permutation invariant loss to separate different instances and group pixels in one obj.
        3) M-S loss to regularize the segmentation level
        4) quantity loss to force predicted value close to integer.
    '''
    def __init__(self, config, class_weights=None, ignore_label=[255]):
        """
        @Param: config -- loss related configurations
                class_weights -- dict with key is the class label and value is the weight
                ignore_label -- ignore class label
        """

        super(LossEvaluate, self).__init__()
        self.config    = config
        self.softmax2d = nn.Softmax2d()

        if class_weights is not None:
            class_weights = [class_weights[ele] for ele in sorted(class_weights.keys()) \
                                                 if ele not in ignore_label]
            class_weights = torch.FloatTensor(class_weights).cuda()

        # loss functions
        self.Binary_loss, self.PI_loss, self.Rfn_loss = None, None, None
        self.MS_loss, self.Quanty_loss, self.Evaluate = None, None, None
        self.setupSingleChannelLosses(class_weights)


    def setupSingleChannelLosses(self, class_weights):
        if 'pi_alpha' in self.config and self.config['pi_alpha']>0:
            if self.config['pi_mode'] == 'sample-list':
                self.PI_loss = PermuInvLoss(class_weights=class_weights,
                                            margin=self.config['pi_margin'],
                                            pi_pairs=self.config['pi_smpl_pairs'],
                                            smpl_wght_en=self.config['pi_smpl_wght_en'],
                                            pos_wght=self.config['pi_pos_wght'],
                                            loss_type=self.config['pi_loss_type'])
            else: # dilate-conv
                self.PI_loss = PermuInvLossDilatedConv(cuda=torch.cuda.is_available(),
                                            margin=self.config['pi_margin'],
                                            pi_ksize=self.config['pi_ksize'],
                                            pi_kcen=self.config['pi_kcen'],
                                            pi_kdilate=self.config['pi_kdilate'],
                                            loss_type=self.config['pi_loss_type'])

        if 'binary_alpha' in self.config and self.config['binary_alpha']>0:
            self.Binary_loss = BinaryLoss(margin=self.config['binary_margin'])

        if 'regul_alpha' in self.config and self.config['regul_alpha']>0:
            self.MS_loss = MumfordShahLoss()

        if 'quanty_alpha' in self.config and self.config['quanty_alpha']>0:
            self.Quanty_loss = QuantityLoss()

        if 'rfn_en' in self.config and self.config['rfn_en']==1:
            self.Rfn_loss = RefineLoss(roi_size=self.config['rfn_roi_size'],
                                       cls_weights=class_weights,
                                       iou_l_thr=self.config['rfn_iou_l_thr'],
                                       iou_h_thr=self.config['rfn_iou_h_thr'],
                                       pos_weight=self.config['rfn_pos_wght'])

        if self.Rfn_loss is not None and 'eval_en' in self.config and self.config['eval_en']:
            self.Evaluate = Evaluate(size_thrs=self.config['eval_size_thrs'],
                                     cls_score_thr=self.config['eval_cls_score_thr'],
                                     iou_thr  =self.config['eval_iou_thr'],
                                     eval_classes=self.config['eval_classes'])
        return


    def stableSoftmax(self, logits):
        max_logits = logits.max(dim=1, keepdim=True)[0]
        max_logits.require_grad = False
        return self.softmax2d(logits - max_logits)

    @torch.no_grad()
    def create_global_slope_plane(self, ht, wd):
        '''
        return a tensor in shape [1, 1, ht, wd] with value = row+col
        '''
        slopeX = np.cumsum(np.ones([ht, wd]), axis=1)
        slopeY = np.cumsum(np.ones([ht, wd]), axis=0)
        plane  = slopeX + slopeY
        return torch.FloatTensor(plane[np.newaxis, np.newaxis, ...])

    @torch.no_grad()
    def mapLocalMask2Global(self, batch_size, loc_mask_logits, loc_cls_logits,
                                    loc_rois, fht, fwd):
        '''
        @Func:
        @Param: batch_size -- batch size
                loc_mask_logits -- tensor in size [N, 1, roi_ht, roi_wd]
                loc_cls_logits -- tensor in size [N, num_classes]
                loc_rois -- tensor in size [N, 5], as (bk, x0,y0,x1,y1)
                fht / fwd -- height / width of the full image
        @Output: preds_mask -- [N, 1, ht, wd]
                 preds_cls --
        '''
        # local to global
        preds_mask, preds_cls, obj_cnts = [None]*batch_size, [None]*batch_size, [0]*batch_size
        N = loc_mask_logits.size(0)
        for k in range(N):
            bk, x0, y0, x1, y1 = loc_rois[k].int()
            nht, nwd = y1-y0+1, x1-x0+1
            tmp = torch.zeros([1, 1, fht, fwd])
            tmp[:,:,y0:y1+1, x0:x1+1] = F.interpolate(nn.Sigmoid()(loc_mask_logits[k:k+1]),
                                                      size=[nht, nwd],
                                                      mode='bilinear',
                                                      align_corners=True)
            obj_cnts[bk] += 1
            if preds_mask[bk] is None:
                preds_mask[bk] = [tmp>0.5]
                preds_cls[bk]  = [loc_cls_logits[k][None, None, :]]
            else:
                preds_mask[bk].append(tmp>0.5)
                preds_cls[bk].append(loc_cls_logits[k][None, None, :])

        # batch stack
        max_cnt = max(obj_cnts)
        for k in range(batch_size):
            if obj_cnts[k] < max_cnt:
                comp_mask = torch.zeros([1, max_cnt-obj_cnts[k], fht, fwd], dtype=torch.bool)
                preds_mask[k].append(comp_mask)
                comp_cls = torch.zeros([1, max_cnt-obj_cnts[k], loc_cls_logits.size(1)])
                preds_cls[k].append(comp_cls)
            preds_mask[k] = torch.cat(preds_mask[k], dim=1)
            preds_cls[k] = torch.cat(preds_cls[k], dim=1)
        preds_mask = torch.cat(preds_mask, dim=0)
        preds_cls = torch.cat(preds_cls, dim=0)

        return {'mask': preds_mask,
                'cls': preds_cls}


    def forward(self, preds, targets, preds_rfn=None, target_boxes=None):
        ''' Compute loss to train the network and report the evaluation metric
        Params: preds -- instance label prediction
                targets -- tensor in [bs, ch, ht, wd] with full GT objects.
                preds_rfn -- dict for refine-classify prediction
                target_boxes -- list of GT object bboxes with [x0,y0,x1,y1,cls_id-1]
        '''
        # prepare data
        bs, ch, ht, wd = preds.size()
        target_ids = torch.zeros(bs, targets.size(1)-1, dtype=torch.int)
        for b in range(bs):
            target_ids[b] = target_boxes[b][:,-1].int()

        # compute loss
        targets_rs = F.interpolate(targets, size=[ht, wd], mode='bilinear', align_corners=True)
        ret = self.process_step_singleChannel(preds, targets_rs, preds_rfn, target_ids)

        # tfboard visual tensors
        ret['preds_0'] = preds  # used for compute gradient
        ret['preds']   = F.relu(preds)
        _, ret['gts']  = targets[:,:-1,:,:].max(axis=1, keepdim=True)
        ret['wghts']   = targets[:,-1:,:,:]

        return ret

    def process_step_singleChannel(self, preds, targets,
                                         preds_rfn=None, target_ids=None,
                                         is_training=True):
        '''
        @Params: preds -- tensor in size [bs, 1, ht, wd]
                 targets -- tensor in [bs, ch, ht, wd] with full GT objects.
                 preds_rfn -- dict from refineNet output
                 target_ids -- tensor in [bs, ch], GT categoryID for each target
                 is_training -- if False, only run Evaluate
        '''
        ret = {}
        weights = targets[:, -1:, :, :]
        _, gts  = targets[:, :-1, :, :].max(axis=1, keepdim=True)
        gts_onehot = (targets[:, :(gts.max()+1), :, :]>0.5).int()
        target_ids = target_ids[:, :(gts.max()+1)]

        bs, _, ht, wd = preds.size()
        if 'glb_trend_en' in self.config and self.config['glb_trend_en']==1:
            plain = self.create_global_slope_plane(ht, wd)
            preds = preds+plain.cuda()

        # compute loss 1
        if is_training:
            if self.Binary_loss is not None:
                loss = self.Binary_loss(preds, gts, weights=weights)
                ret['binary'] = loss * self.config['binary_alpha']

            # for other losses, perform Relu on predicted labels
            preds_relu = F.relu(preds)
            if self.PI_loss is not None:
                pi_weights= None if self.config['pi_smpl_wght_en']>0 else weights
                if self.config['pi_mode'] == 'sample-list':
                    loss = self.PI_loss(preds_relu, gts,
                                        target_ids=target_ids,
                                        weights=pi_weights,
                                        BG=self.config['pi_hasBG'])
                else:
                    loss = self.PI_loss(preds_relu, gts.float(),
                                        weights=pi_weights,
                                        BG=self.config['pi_hasBG'])
                if loss is not None:
                    ret['pi'] = loss['loss'] * self.config['pi_alpha']
                    ret['eval_pi0'] = loss['eval_pi0']
                    ret['eval_pi1'] = loss['eval_pi1']

            if self.Quanty_loss is not None:
                loss = self.Quanty_loss(preds_relu)
                if loss is not None:
                    ret['quanty'] = loss * self.config['quanty_alpha']

            if self.MS_loss is not None:
                loss = self.MS_loss(preds_relu)
                if loss is not None:
                    ret['regul'] = loss * self.config['regul_alpha']

            if self.Rfn_loss is not None:
                rfn_loss = self.Rfn_loss(preds_rfn, gts_onehot, target_ids)
                ret['rfn_iou'] = rfn_loss['iou_loss']*self.config['rfn_iou_alpha']
                ret['rfn_cls'] = rfn_loss['cls_loss']*self.config['rfn_cls_alpha']
                ret['rfn_seg'] = rfn_loss['seg_loss']*self.config['rfn_seg_alpha']

        # evaluation
        if self.Evaluate is not None:
            local_rois = torch.cat(preds_rfn['obj_bboxes'], dim=0)[:, :5]
            preds_glb = self.mapLocalMask2Global(bs,
                                                 preds_rfn['obj_masks'],
                                                 preds_rfn['cls_logits'],
                                                 local_rois, ht, wd)
            evalV = self.Evaluate(preds_glb['mask'],
                                  gts_onehot[:, 1:, :, :],
                                  preds_glb['cls'],
                                  target_ids[:, 1:])
            ret['eval_prec'] = evalV['prec']
            ret['eval_rec'] = evalV['rec']
            ret['eval_acc'] = evalV['acc']

        return ret

