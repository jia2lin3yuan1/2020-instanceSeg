import torch
from torch import nn

class Evaluate(nn.Module):
    ''' This class computes IoU to evaluate the predictions' quality @ Recall @ Precision @ Acc
    '''
    def __init__(self, size_thrs=1.0, cls_score_thr=0.5, iou_thr=0.5,
                        meter_range=500, eval_classes=None):
        super(Evaluate, self).__init__()
        self.size_thrs     = size_thrs
        self.cls_score_thr = cls_score_thr
        self.iou_thr       = iou_thr

        self.meter_range  = meter_range
        self.eval_classes = eval_classes
        self.softmax      = nn.Softmax(dim=-1)

        self.meter = 0
        # pred / GT:   1,    0
        #        1 :  TP,   FP
        #          :
        #        0 :  FN    TN
        self.prec = None  # tp / (tp+fp)
        self.rec  = None  # tp / (tp + fn)
        self.acc  = None  # (tp)/(tp+fp+fn)
        #self.acc  = None  # (tp+tn)/(tp+fp+tn+fn)

    @torch.no_grad()
    def forward(self, pred_masks, target_masks, pred_logits, target_clsIds):
        '''
        Params:
            pred_masks -- [bs,ch, ht, wd]
            target_masks -- [bs,ch', ht, wd]  # only FG objects
            pred_clsId -- [bs, ch, num_class], from classification branch
            target_clsIds -- [bs, ch'], class ids from target_masks
        '''
        bs, p_ch      = pred_masks.size()[:2]
        gt_ch = target_masks.size(1)

        # compute IoU
        pred_m_1d   = (pred_masks.view(bs, p_ch, -1)>0.5).float() # [bs, ch, N]
        target_m_1d = target_masks.view(bs, gt_ch, -1).permute(0,2,1).float() # [bs, N, ch']
        intp        = torch.matmul(pred_m_1d, target_m_1d) # [bs, ch, ch']
        pred_sum    = pred_m_1d.sum(axis=-1, keepdim=True) # [bs, ch, 1]
        target_sum  = target_m_1d.sum(axis=1, keepdim=True) # [bs, 1, ch']
        union       = pred_sum + target_sum - intp # [bs, ch, ch']
        iou         = intp / (union + 1e-2)

        # confidence
        pred_score,  _ = self.softmax(pred_logits).max(axis=-1) # [bs, ch]
        _, pred_clsIds = pred_logits.max(axis=-1)  # [bs, ch]
        pred_clsIds    = pred_clsIds.int()

        # compute tp/fp
        tp, fp = 0., 0.
        for b in range(bs):
            idx = pred_clsIds[b].nonzero()
            if len(idx) == 0:
                continue
            else:
                _, sort_idx = pred_score[b][idx].sort(axis=0, descending=True)
                for k in sort_idx:
                    pk = idx[k] # index of the selected prediction

                    # ignore small predictions
                    if pred_sum[b, pk] < self.size_thrs or pred_score[b, pk] < self.cls_score_thr:
                        continue
                    # if the predicted category is not in eval_classes
                    if self.eval_classes is not None and pred_clsIds[b, pk] not in self.eval_classes:
                        continue

                    # map the candidate to a un-mapping GT object
                    map_iou, map_gk = iou[b, pk].max(axis=-1)
                    cond_iou = map_iou>=self.iou_thr
                    cond_cls  = pred_clsIds[b, pk] == target_clsIds[b, map_gk]
                    if cond_iou and cond_cls:
                        tp += 1.
                        iou[b, :, map_gk] = 0.
                    else:
                        fp += 1.

        # compute precision / recall / accuracy
        if self.eval_classes is not None:
            eff_gt_cls = [ele in self.eval_classes for ele in target_clsIds.flatten()]
            tot_target = torch.tensor(eff_gt_cls).float().sum()
        else:
            tot_target = (target_clsIds>0).float().sum()
        # fn = tot_gts - tp

        precision = torch.tensor(tp / (tp + fp + 1e-3))
        recall    = torch.tensor(tp / (tot_target + 1e-3))
        accuracy  = torch.tensor(tp /(tot_target + fp +1e-3))

        if self.meter==0:
            self.prec, self.rec, self.acc = precision, recall, accuracy
        else:
            if tp+fp > 0 or tot_target > 0:
                self.meter = (self.meter+1) if self.meter < self.meter_range else self.meter
                self.prec = (self.meter*self.prec+ precision)/(self.meter+1.)
                self.rec  = (self.meter*self.rec + recall)/(self.meter+1.)
                self.acc  = (self.meter*self.acc + accuracy)/(self.meter+1.)

        return {'prec': self.prec,
                'rec': self.rec,
                'acc': self.acc }


