import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from layers.modules.instance_loss import BinaryLoss, PermuInvLoss, PermuInvLossDilatedConv
from layers.modules.regularize_loss import QuantityLoss, MumfordShahLoss

class LossEvaluate(nn.Module):
    '''
    compute loss:
        1) binary loss to separate BG and FG
        2) permutation invariant loss to separate different instances and group pixels in one obj.
        3) M-S loss to regularize the segmentation level
        4) if predict in 1 channel, use quantity loss to force predicted value close to integer.
           else, use iou-loss to force predicted value close to 1 on corresponding GT.
    '''
    def __init__(self, config, class_weights=None, ignore_label=[255]):
        """
        @Param: config -- loss related configurations
                class_weights -- dict with key is the class label and value is the weight
                ignore_label -- ignore class label
        """
        super(LossEvaluate, self).__init__()
        self.config = config
        self.fg_stCH = fg_stCH
        self.softmax2d = nn.Softmax2d()
        if class_weights is not None:
            class_weights = [class_weights[ele] for ele in sorted(class_weights.keys()) if ele not in ignore_label]
            class_weights = torch.FloatTensor(class_weights).cuda()


        # loss functions
        self.Binary_loss, self.MS_loss, self.PI_loss = None, None, None
        self.Quanty_loss = None
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
                                            loss_type=self.config['pi_loss_type'],
                                            pi_ksize=self.config['pi_ksize'],
                                            pi_kcen=self.config['pi_kcen'],
                                            pi_kdilate=self.config['pi_kdilate'])

        if 'binary_alpha' in self.config and self.config['binary_alpha']>0:
            self.Binary_loss = BinaryLoss(margin=self.config['binary_margin'])

        if 'regul_alpha' in self.config and self.config['regul_alpha']>0:
            self.MS_loss = MumfordShahLoss()

        if 'quanty_alpha' in self.config and self.config['quanty_alpha']>0:
            self.Quanty_loss = QuantityLoss()

    @torch.no_grad()
    def create_global_slope_plane(self, ht, wd):
        '''
        return a tensor in shape [1, 1, ht, wd] with value = row+col
        '''
        slopeX = np.cumsum(np.ones([ht, wd]), axis=1)
        slopeY = np.cumsum(np.ones([ht, wd]), axis=0)
        plane  = slopeX + slopeY
        return torch.FloatTensor(plane[np.newaxis, np.newaxis, ...])

    def forward(self, preds, targets, target_boxes=None):
        ''' Compute loss to train the network and report the evaluation metric
        Params: preds -- list of instance prediction in different scale, from FPN.
                targets -- tensor in [bs, ch, ht, wd] with full GT objects.
                target_boxes -- list of GT object bboxes with [x0,y0,x1,y1,cls_id-1]
        '''
        bs, ch, _, _ = preds[0].size()
        target_ids = torch.zeros(bs, targets.size(1)-1, dtype=torch.int)
        for b in range(bs):
            target_ids[b] = target_boxes[b][:,-1].int()

        for k in range(len(preds)):
            if pred_logits is not None:
                tmp_ret = self.process_onescale(preds[k], targets, pred_logits[k], target_ids)
            else:
                tmp_ret = self.process_onescale(preds[k], targets, target_ids)

            if k == 0:
                ret = tmp_ret
                ret['preds'] = preds[0]
                _, ret['gts']= targets[:,:-1,:,:].max(axis=1, keepdim=True)
                ret['wghts'] = targets[:,-1:,:,:]
            else:
                for key in tmp_ret:
                    ret[key] += tmp_ret[key]

        return ret

    @torch.no_grad()
    def resize_GT(self, targets, nht, nwd):
        """
        @Func: resize GT make sure BG channels has no overlap 1.
             if bg_ch == 1, perform bilinear intepolation on each channel, and >0.5 to obtain binary mask
             if bg_ch > 1, perform nearest intepolation on BG channels by combining bg channels into one channel
                           perform bilinear intepolation on FG channels.
        """
        if self.fg_stCH == 1:
            gts_rs  = F.interpolate(targets, size=[nht, nwd], mode='bilinear', align_corners=True)
        else:
            _, targets_bg = targets[:, :-1, :, :].max(axis=1, keepdim=True)
            targets_bg[targets_bg >=self.fg_stCH] = self.fg_stCH
            gts_rs_bg  = F.interpolate(targets_bg.float(), size=[nht, nwd], mode='nearest') # [bs,ch, ht, wd]
            gts_rs_bg  = torch.eye(self.fg_stCH+1)[gts_rs_bg[:,0,:,:].long(), :] # [bs, ht, wd, ch]
            gts_rs_bg = gts_rs_bg.permute([0,3,1,2])[:, :self.fg_stCH, :, :] #[bs, ch, ht, wd]

            targets_fg = targets[:, self.fg_stCH:, :, :]
            gts_rs_fg  = F.interpolate(targets_fg,
                                       size=[nht, nwd],
                                       mode='bilinear', align_corners=True)
            gts_rs = torch.cat([gts_rs_bg, gts_rs_fg], axis=1)

        return gts_rs


    def process_onescale(self, preds, targets, target_ids=None):
        ''' Compute loss to train the network and report the evaluation metric
        Params: preds -- tensor for instance prediction.
                targets -- tensor in [bs, ch, ht, wd] with full GT objects.
                target_ids -- tensor in [bs, ch], GT categoryID for each target
        '''
        bs, ch, ht, wd = preds.size()
        # prepare data
        gts_rs    = self.resize_GT(targets, ht, wd)
        weights_0 = gts_rs[:, -1:, :, :]
        preds_0 = preds
        _, gts_0 = gts_rs[:, :-1, :, :].max(axis=1, keepdim=True)
        gts_onehot = (gts_rs[:, :(gts_0.max()+1), :, :]>0.5).int()

        if 'glb_trend_en' in self.config and self.config['glb_trend_en']==1:
            plain = self.create_global_slope_plane(ht, wd)
            preds_0 = preds_0+plain.cuda()

        # compute loss
        ret = {}
        ret['preds_0'] = preds_0

        if self.Binary_loss is not None:
            loss = self.Binary_loss(preds_0, gts_0, weights=weights_0)
            ret['binary'] = loss * self.config['binary_alpha']

        preds_1 = F.relu(preds_0)
        if self.Quanty_loss is not None:
            loss = self.Quanty_loss(preds_1)
            ret['quanty'] = loss * self.config['quanty_alpha']

        if self.MS_loss is not None:
            loss = self.MS_loss(preds_1)
            ret['regul'] = loss * self.config['regul_alpha']

        if self.PI_loss is not None:
            if self.config['pi_mode'] == 'sample-list':
                loss = self.PI_loss(preds_1, gts_0,
                                    target_ids=target_ids,
                                    weights= None if self.config['pi_smpl_wght_en']>0 else weights_0,
                                    BG=self.config['pi_hasBG'])
            else:
                loss = self.PI_loss(preds_1, gts_0.float(),
                                    weights= None if self.config['pi_smpl_wght_en']>0 else weights_0,
                                    BG=self.config['pi_hasBG'])
            if loss is not None:
                ret['pi'] = loss['loss'] * self.config['pi_alpha']
                ret['eval_pi0'] = loss['eval_pi0']
                ret['eval_pi1'] = loss['eval_pi1']

        return ret
