#copyright (c) 2021-present, jialin yuan@Deep Vision Group
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from layers.modules.classify_loss import CrossEntropyLoss
from layers.modules.classify_loss import BinaryCrossEntropyLoss
from torchvision.ops import roi_align


def adjust_smooth_l1_loss(y_pred, theta=0.1):
    # small gradient when close to 0, constant gradient in large value zone
    less_grad_factor = 1./(2*theta)
    less_loss_bias   = less_grad_factor * theta**2
    less_than_theta  = (y_pred < theta).float()
    loss = (less_than_theta*(y_pred**2)*less_grad_factor) + \
           (1-less_than_theta)*(y_pred-theta + less_loss_bias)
    return loss


class RoIAlign(nn.Module):
    def __init__(self, out_size=(28,28), scale=1.0, aligned=True):
        '''
        @Param: scale -- scale that map feature coordinates to bboxes' coordinates.
                        ( e.x. feature (8, 8) map to box (16,16) with scale 0.5)

        '''
        super(RoIAlign, self).__init__()
        self.out_size = out_size
        self.scale = scale
        self.aligned = aligned

    def forward(self, input, bboxes):
        '''
        @Param: input--tensor of size [bs, ch, ht, wd]
                bboxes -- float tensor of size [N,,,, 5] with [b_idx, x0, y0, x1, y1], or list of
                        tensor with size [n_i, 4], with each one corresponding to one batch example
        @Output: tensor of size [N, ch, out_size[0], out_size[1]]

        '''
        rois = roi_align(input.float(), bboxes.float(), spatial_scale=self.scale,
                         output_size=self.out_size) # , aligned=self.aligned)
        '''
        current torch version is 1.3.0, its roi_align has no argument aligned
        '''
        return rois


class RefineLoss(nn.Module):
    '''This class computes the loss for DVIS when it outputs single channel prediction,
    the loss includes iou_loss, classify_loss, mask_loss
    '''
    def __init__(self, roi_size=(28,28), scale=1.0, cls_weights=None,
                        iou_l_thr=0.1, iou_h_thr=0.3, pos_weight=1.0):
        super(RefineLoss, self).__init__()
        self.roi_size    = roi_size
        self.cls_weights = cls_weights
        self.iou_l_thr   = iou_l_thr
        self.iou_h_thr   = iou_h_thr

        self.RoIPooling  = RoIAlign(out_size=roi_size, scale=scale)
        self.CE_loss_cls = CrossEntropyLoss(weight=cls_weights, reduction='none')
        self.BCE_loss_seg = BinaryCrossEntropyLoss(reduction='none',
                                                   pos_weight=torch.tensor([pos_weight]))

    def computeGTfromTarget(self, pred_mask, pred_labels, rois,
                                   target_onehot, target_ids, correct_cls=False):
        '''
        @Param:
            pred_mask -- tensor, binary mask from refineNet's segment head, in size [N, 1, ht, wd]
            pred_labels -- list (# meanshift bandwidth) of list (# batch size),
                            for each element is a onehot label tensor in size [N, ht, wd]
            rois -- list of rois in size [N, 7], as (bs_idx, x0,y0,x1,y1,label_idx, real_label)
            target_onehot -- tensor, in size [bs, C, ht', wd']. BG is in 1st channel
            target_ids -- tensor, cls_ids in size [C]. the 1st one is BG
    @Output:
            target cls, iou, mask for each roi
        '''
        # remove BG from input targeg
        if target_onehot.size(1)>=2:
            targets  = target_onehot[:, 1:, :, :] # each channel represents for one object
            target_ids = target_ids[:, 1:]
        else:
            # there is no object from GT
            targets = torch.zeros_like(target_onehot)
            target_ids = target_ids

        # statistical over target_ids and prepare target rois
        bs, gt_cnt, _, _ = targets.size()
        target_size = targets.view(bs, gt_cnt, -1).sum(axis=-1)
        target_rois = self.RoIPooling(targets, torch.cat(rois, dim=0)[:, :5]) # [N, C, roi_size]
        target_rois = (target_rois>=0.5).float()

        # find expected category for candidates
        cnt  = -1
        obj_weights, ori_mask = [], []
        target_cls, target_mask, target_iou = [], [], []
        for labelImgs, bboxes in zip(pred_labels, rois):
            for k in range(bboxes.size(0)):
                # compute iou to map target
                bk, x0, y0, x1, y1, labelVal = bboxes[k, :-1].int()
                pred   = (labelImgs[bk][labelVal, y0:y1, x0:x1]).reshape(1, -1) # [1, h*w]
                gt     = targets[bk, :, y0:y1, x0:x1].reshape(gt_cnt, -1) # [C, h*w]
                interp = (pred * gt).sum(axis=-1)
                iou    = interp/(target_size[bk]+pred.sum(axis=-1)-interp+1.0)
                iou_max, gt_ch = iou[:,None].max(axis=0)

                # construct targets for computing loss
                cnt += 1
                if iou_max >= self.iou_h_thr:
                    cls = target_ids[bk, gt_ch]
                    wght = self.cls_weights[cls.long()] + self.cls_weights[0]
                    mask = target_rois[cnt, gt_ch]
                elif (iou_max > 0 and iou_max < self.iou_l_thr) or \
                     (k == bboxes.size(0)-1 and len(target_cls)<2):
                    cls  = torch.zeros(1, dtype=torch.int32)
                    wght = self.cls_weights[cls.long()]
                    mask = torch.zeros(self.roi_size)[None, :, :]
                else:
                    cls  = torch.zeros(1, dtype=torch.int32)
                    wght = torch.zeros(1, dtype=torch.float32)
                    mask = torch.zeros(self.roi_size)[None, :, :]

                # collect data
                rs_ori_mask = F.interpolate(labelImgs[bk][labelVal, y0:y1, x0:x1][None, None, :, :],
                                            size=self.roi_size,
                                            mode='bilinear',
                                            align_corners=True)[0] #[1, 28, 28]
                ori_mask.append(rs_ori_mask)
                target_cls.append(cls)
                target_mask.append(mask)
                target_iou.append(iou_max)
                obj_weights.append(wght)

        # stack to tensor
        target_cls     = torch.stack(target_cls)
        target_mask    = torch.stack(target_mask)
        target_iou     = torch.stack(target_iou)
        obj_weights    = torch.stack(obj_weights)
        ori_mask       = torch.stack(ori_mask)

        # correct target cls w.r.t. refineNet output
        if correct_cls:
            pred_cnt       = pred_mask.size(0)
            pred_mask_1d   = pred_mask.reshape(pred_cnt, 1, -1)
            target_rois_1d = target_rois.reshape(pred_cnt, gt_cnt, -1)
            intp_rois      = (pred_mask_1d * target_rois_1d).sum(axis=-1)
            union_rois     = target_rois_1d.sum(axis=-1)+pred_mask_1d.sum(axis=-1)-intp_rois + 1.0
            rfn_iou        = intp_rois/union_rois
            target_cls     = target_cls * (rfn_iou > self.iou_l_thr)

        return {'iou': target_iou,
                'cls': target_cls,
                'mask': target_mask,
                'weight': obj_weights,
                'ori_mask': ori_mask } # predicted mask from proto-branch

    def forward(self, preds, target_onehot, target_ids, mask_thr=0.5):
        '''
        @Param:
            preds -- dictionary, output from refineNet
            target_onehot -- tensor in size [bs, N', ht', wd']. BG is in 1st channel
            target_ids -- tensor for cls_ids in size [N']. the 1st one is BG
        '''
        mask_logits = preds['obj_masks'] #[N, 1, ht, wd]
        cls_logits  = preds['cls_logits'] #[N, num_cls]
        iou_logits  = preds['iou_scores'] #[N, 1]

        if self.cls_weights is None:
            self.cls_weights = [1.0] * cls_logits.size(1)

        # match GT
        with torch.no_grad():
            pred_prob   = nn.Sigmoid()(mask_logits)
            gts = self.computeGTfromTarget(pred_prob>mask_thr,
                                           preds['labelI'], preds['obj_bboxes'],
                                           target_onehot, target_ids)
        # compute loss
        iou_diff = torch.abs(preds['iou_scores']-gts['iou'])
        iou_loss = adjust_smooth_l1_loss(iou_diff) #[N, 1]
        cls_loss = self.CE_loss_cls(preds['cls_logits'][:, :, None], gts['cls'].long()) # [N,1]
        seg_loss = self.BCE_loss_seg(preds['obj_masks'], gts['mask']) #[N, 1, ht, wd]

        keep = gts['weight'].nonzero()[:, 0]
        if keep.size(0)<4:
            keep = torch.FloatTensor([k for k in range(gts['weight'].size(0))]).long()

        if False:
            bs = target_onehot.size(0)
            preds_all_labelI = [preds['labelI'][0][k].max(axis=0)[1] for k in range(bs)]
            preds_all_labelI = torch.cat(preds_all_labelI, axis=0).cpu().detach().numpy()
            target_all_labelI = [target_onehot[k].max(axis=0)[1] for k in range(bs)]
            target_all_labelI = torch.cat(target_all_labelI, axis=0).cpu().detach().numpy()

            vis_mapped_GT(preds_all_labelI, target_all_labelI,
                          pred_prob.cpu().detach().numpy(),
                          gts['mask'].cpu().detach().numpy(),
                          gts['iou'][:,0].cpu().detach().numpy(),
                          gts['cls'][:,0].cpu().detach().numpy(),
                          torch.cat(preds['obj_bboxes'], dim=0)[:, :5].int().cpu().detach().numpy(),
                          keep.cpu().detach().numpy())
            import pdb; pdb.set_trace()

        # apply weights
        seg_loss = seg_loss.mean(axis=-1).mean(axis=-1) #[N,1]
        wght_sum = gts['weight'].sum()+1e-4  # divide 0 protect
        iou_loss = (iou_loss*gts['weight']).sum()/wght_sum
        cls_loss = (cls_loss*gts['weight']).sum()/wght_sum
        seg_loss = (seg_loss*gts['weight']).sum()/wght_sum

        ret_dict = {'iou_loss': iou_loss,
                    'cls_loss': cls_loss,
                    'seg_loss': seg_loss}
        return ret_dict

def vis_mapped_GT(pred_all_label, target_all_label,
                  pred_rois_prob, target_rois_mask, ious, clses, bboxes, keep=None):
    '''
    @Param: pred_all_label | target_all_label -- [ht, wd]
            pred_rois_prob | target_rois_mask -- [N, 1, ht, wd]
            ious | clses -- [N, 1]
            bboxes -- [N, 4]
            keep -- a list with idx between [0, N)]
    '''
    from matplotlib import pyplot as plt
    if keep is not None:
        pred_rois_prob   = pred_rois_prob[keep]
        target_rois_mask = target_rois_mask[keep]
        ious, clses, bboxes = ious[keep], clses[keep], bboxes[keep]

    step = 4
    N    = pred_rois_prob.shape[0]
    for k in range(0, N, step):
        col = min(N-k, step)
        print('iou are: ', ious[k: k+step])
        print('cls are: ', clses[k: k+step])
        print('bboxe are: \n', bboxes[k: k+step])
        fig, ax = plt.subplots(col+1, 2)
        ax[0,0].imshow(pred_all_label)
        ax[0,1].imshow(target_all_label)

        for i in range(1, col+1):
            ax[i, 0].imshow(pred_rois_prob[k+i-1, 0])
            ax[i, 1].imshow(target_rois_mask[k+i-1, 0])
        plt.show()
        plt.close()
