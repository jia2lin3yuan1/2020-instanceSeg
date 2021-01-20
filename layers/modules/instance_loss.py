import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from layers.modules.python_func import create_pairwise_conv_kernel

def adjust_smooth_l1_loss(y_pred, theta=0.1):
    # small gradient when close to 0, constant gradient in large value zone
    less_grad_factor = 1./(2*theta)
    less_loss_bias   = less_grad_factor * theta**2
    less_than_theta  = (y_pred < theta).float()
    loss = (less_than_theta*(y_pred**2)*less_grad_factor) + \
           (1-less_than_theta)*(y_pred-theta + less_loss_bias)
    return loss


def pi_l1_loss(pi_pred, pi_target, margin=1.0, pos_wght=3.0):
    '''
    Params: pi_pred -- float tensor in size [N, N]
            pi_target -- float tensor in size [N, N], with value {0, 1}, which
                        ** 1 for pixels belonging to different object
                        ** 0 for pixels belonging to same object
            margin/pos_wght -- float
    '''
    loss_pi_1 = adjust_smooth_l1_loss(F.relu(margin-pi_pred))
    loss_pi_0 = adjust_smooth_l1_loss(pi_pred)

    loss = loss_pi_1*pi_target*pos_wght + loss_pi_0*(1.0-pi_target)
    return loss

def pi_exp_loss(pi_pred, pi_target, margin=1.0, pos_wght=3.0):
    '''
    @ refering paper 'semantic instance segmentation via deep matric learning
    Params: pi_pred -- float tensor in size [N, N]
            pi_target -- float tensor in size [N, N], with value {0, 1}
                        ** 1 for pixels belonging to different object
                        ** 0 for pixels belonging to same object
            margin/pos_wght -- float
    '''
    pi_diff = pi_pred
    pi_sims = 2.0*(margin-torch.sigmoid(pi_diff))

    loss = -(pi_target       * torch.log(1.0-pi_sims)*pos_wght + \
             (1.0-pi_target) * torch.log(pi_sims))
    return loss


class BinaryLoss(nn.Module):
    '''This class computes the Binary loss to force BG pixels close to 0 and FG pixels far away.
    '''
    def __init__(self, margin=2.0):
        """
        margin: minimum distance between FG/BG if prediction with 1 channel
        """
        super(BinaryLoss, self).__init__()
        self.margin = margin

    def forward(self, preds, targets, weights=None):
        '''
        Params:
            preds/targets: [bs, 1, ht, wd]
            weights:[bs, 1, ht, wd]
        '''
        # preds is the instance label value
        preds_0, targets_0 = preds, targets
        isFG   = (targets_0>0.5).float()
        loss_0 = adjust_smooth_l1_loss(F.relu(preds))
        loss_1 = adjust_smooth_l1_loss(F.relu(self.margin-preds))
        loss = loss_0*(1.0-isFG) + loss_1*isFG

        if weights is not None:
            loss = torch.mul(loss, weights)

        return loss.mean().float()


class PermuInvLoss(nn.Module):
    '''This class compute the permutation-invariant loss between the targets and the predictions.
        The pixels used to compute pi_loss is sampled on each objects
        It encourages pixels in same instance close to each other,
                      pixels from different instances far away from each other.

    '''
    def __init__(self, class_weights=None, margin=1.0, pi_pairs=4096, avg_num_obj=8,
                       smpl_wght_en=0.0, pos_wght=3.0, loss_type='l1'):
        super(PermuInvLoss, self).__init__()
        self.class_weights = class_weights
        self.margin    = margin
        self.pi_pairs  = pi_pairs
        self.avg_num_obj = avg_num_obj
        self.smpl_wght_en   = smpl_wght_en
        self.pos_wght  = pos_wght
        self.loss_type = loss_type

    @torch.no_grad()
    def sampling_over_objects(self, targets, target_classes, BG=0):
        '''
        Params:
            targets -- tensor in [N, ch] with integers values.
            target_classes -- if not None, in size [M], M is No. of Objs in targets.
            BG -- if True, BG is counted as one object.
        '''
        eff_idx, smpl_wght = [], []

        N, ch   = targets.size()
        targets_onehot = targets if ch>1 else torch.eye(targets.max()+1)[targets[:,0].long(),:]
        cnts    = targets_onehot.sum(axis=0)
        num_obj = (cnts>0).sum() if BG else (cnts[1:]>0).sum()

        if num_obj > 0:
            # sample over each object
            avg     = self.pi_pairs//num_obj
            for k, cnt in enumerate(cnts):
                if cnt==0 or (BG==0 and k<1):
                    continue

                # sample index on current object
                idx = (targets_onehot[:, k]==1).nonzero()
                perm = torch.randperm(idx.size(0))
                cur_sel = idx[perm][:avg]
                smpl_size = torch.FloatTensor([cur_sel.size(0)]).cuda()
                obj_wght = torch.pow(self.pi_pairs/(smpl_size + 1.), 1./3)
                if target_classes is not None and self.class_weights is not None:
                    obj_wght += self.class_weights[target_classes[k]]

                # add into the whole stack.
                eff_idx.extend(cur_sel)
                smpl_wght.extend(torch.ones(cur_sel.size(0), 1, dtype=torch.float)*obj_wght)
        return eff_idx, smpl_wght

    def forward(self, preds, targets, target_ids=None, weights=None, BG=False, sigma=1e-3):
        ''' Compute the permutation invariant loss on pixel pairs if both pixels are in the instances.
        Params:
            preds/targets -- [bs, 1, ht, wd]. here,
                            ** preds is logits from Relu(), targets are integer labels
            weights -- [bs, 1, ht, wd]
            BG -- if True, treat BG as one instance.
                | if False, don't sample point on BG pixels,
                            and compute difference only on FG
        '''
        all_loss = []
        bs, ch, ht, wd = targets.size()

        # reshape
        preds_1D = preds.view(bs, ch, -1).permute(0, 2, 1) # in size [bs, N, 1]
        targets_1D = targets.view(bs, ch, -1).permute(0, 2, 1) # in size [bs, N, 1]
        if weights is not None:
            weight_1D = weights.view(bs, -1, 1)

        # compute loss for each sample
        all_loss = []
        eval_pi0, eval_pi1 = [], []
        for b in range(bs):
            smpl_idx, smpl_wght = self.sampling_over_objects(targets_1D[b], target_ids[b], BG=BG)
            if len(smpl_idx) == 0:
                continue

            # compute pairwise differences over pred/target/weight
            smpl_pred = preds_1D[b][smpl_idx,:]
            smpl_target = targets_1D[b][smpl_idx,:].float()

            # l1 distance = abs(x1-x2)
            pi_pred = torch.clamp(torch.abs(smpl_pred-smpl_pred.permute(1,0)), 1e-4, 5) # [N, N]
            pi_target = (torch.abs(smpl_target - smpl_target.permute(1,0))>0.5).float()

            if weights is not None:
                smpl_weight = weight_1D[b][smpl_idx, :]
                pi_obj_wght = smpl_weight + smpl_weight.permute(1,0)
            elif self.smpl_wght_en>0.0:
                smpl_wght   = torch.stack(smpl_wght) #[N, 1]
                pi_obj_wght = (smpl_wght + smpl_wght.permute(1,0))
            else:
                pi_obj_wght = None

            # compute loss
            if self.loss_type == 'l1':
                loss = pi_l1_loss(pi_pred, pi_target, self.margin, self.pos_wght)
            else: # 'DM-exp'
                loss = pi_exp_loss(pi_pred, pi_target, self.margin, self.pos_wght)
            if pi_obj_wght is not None:
                loss = torch.mul(loss, pi_obj_wght)

            loss = loss[loss>sigma]
            if loss.size(0)>0:
                all_loss.append(loss.mean())

            # for evaluation:
            eval_pi0.append((pi_pred*(pi_target==0).float()).mean())
            eval_pi1.append((pi_pred*(pi_target==1).float()).mean())
        if len(all_loss)>0:
            return {'loss': torch.stack(all_loss).mean(),
                    'eval_pi0': torch.stack(eval_pi0).mean(),
                    'eval_pi1': torch.stack(eval_pi1).mean()}
        else:
            return None


class PermuInvLossDilatedConv(nn.Module):
    '''
    in this class, compute the permutation-invariant loss via setting dilated convolution kernel and run convolution.
    '''
    def __init__(self, cuda=True, margin=1.0, pos_wght=3.0, loss_type='l1',
                       pi_ksize=[32, 64], pi_kcen=8, pi_kdilate=2):
        super(PermuInvLossDilatedConv, self).__init__()
        self.cuda       = cuda
        self.margin     = margin
        self.pos_wght   = pos_wght
        self.loss_type  = loss_type

        self.pi_ksize   = pi_ksize
        self.pi_kcen    = pi_kcen
        self.pi_kdilate = pi_kdilate
        self.pi_diff_ke, self.pi_sum_ke = self.customize_pi_kernel()

    def customize_pi_kernel(self):
        '''
        @func: generate kernel for convolution in size [N, 1, kht, kwd]
        '''
        diff_kernel = self.construct_pairwise_kernel(do_diff=True)
        sum_kernel= self.construct_pairwise_kernel(do_diff=False)

        if self.cuda:
            diff_kernel, sum_kernel = diff_kernel.cuda(), sum_kernel.cuda()

        return diff_kernel, sum_kernel

    def construct_pairwise_kernel(self, do_diff=False):
        '''
        @func: generate kernel for convolution in size [N, 1, kht, kwd]
        '''
        kernel_size, kernel_cen, kernel_stride=self.pi_ksize, self.pi_kcen, self.pi_kdilate
        #  py_kernel in shape [in_ch, ht, wd, out_ch]
        kernel = create_pairwise_conv_kernel(kernel_size,
                                             kernel_cen,
                                             dia_stride=kernel_stride,
                                             diff_kernel=do_diff)

        # output tc kernel in [out_ch, in_ch, ht, wd]
        tc_kernel = torch.FloatTensor(kernel)
        tc_kernel = tc_kernel.permute(3, 0, 1, 2)
        return tc_kernel

    @torch.no_grad()
    def compute_pairwise_conv_nograd(self, tensor, kernel):
        """
        @func: compute pairwise relationship
        @param: tensor -- size [bs, 1, ht, wd]
                kernel -- size [N, 1, kht, kwd]

        @output: result in size [N, :]
        """
        kht, kwd       = kernel.shape[2], kernel.shape[3]
        pad_ht, pad_wd = kht//2, kwd//2
        padT = torch.nn.ReplicationPad2d([pad_wd, pad_wd, pad_ht, pad_ht])(tensor)
        out  = F.conv2d(padT, kernel) # [bs, N, ht, wd]
        assert(out.shape[2:] == tensor.shape[2:])

        out  = out.permute(1,0,2,3).reshape(kernel.size(1), -1) # [N, :]
        return out

    def compute_pairwise_conv(self, tensor, kernel):
        """
        @func: compute pairwise relationship
        @param: tensor -- size [bs, 1, ht, wd]
                kernel -- size [N, 1, kht, kwd]

        @output: result in size [N, :]
        """
        kht, kwd       = kernel.shape[2], kernel.shape[3]
        pad_ht, pad_wd = kht//2, kwd//2
        padT = torch.nn.ReplicationPad2d([pad_wd, pad_wd, pad_ht, pad_ht])(tensor)
        out  = F.conv2d(padT, kernel)  # [bs, N, ht, wd]
        assert(out.shape[2:] == tensor.shape[2:])

        out  = out.permute(1,0,2,3).reshape(kernel.size(0), -1) # [N, :]

        return out

    def pi_l1_loss(self, pi_pred, pi_target):
        '''
        Params: pi_pred -- float tensor in size [bs, N, ht, wd]
                pi_target -- float tensor in size [bs, N, ht, wd, N], with value {0, 1}, which
                            ** 1 for pixels belonging to different object
                            ** 0 for pixels belonging to same object
        '''
        loss_pi_1 = adjust_smooth_l1_loss(F.relu(self.margin-pi_pred))
        loss_pi_0 = adjust_smooth_l1_loss(pi_pred)

        loss = loss_pi_1*pi_target*self.pos_wght + loss_pi_0*(1.0-pi_target)
        return loss

    def pi_exp_loss(self, pi_pred, pi_target):
        '''
        @ refering paper 'semantic instance segmentation via deep matric learning
        Params: pi_pred -- float tensor in size [N, N]
                pi_target -- float tensor in size [N, N], with value {0, 1}
                            ** 1 for pixels belonging to different object
                            ** 0 for pixels belonging to same object
        '''
        pi_diff = torch.clamp(torch.abs(pi_pred), 1e-4, 3.0).float()
        pi_sims = 2.0*(1.0-torch.sigmoid(pi_diff))

        loss = -(pi_target       * torch.log(1.0-pi_sims)*self.pos_wght + \
                 (1.0-pi_target) * torch.log(pi_sims))
        return loss

    def forward(self, preds, targets, weights=None, BG=False):
        '''
        @func:
        @params: targets / preds / weights: tensor in shape [bs, 1, ht, wd]
                 pos_wght -- intra pw is positive, from different instances
                 BG -- if False, only take FG pixels as the central pixel
        '''
        bs, _, ht, wd = targets.size()


        # compute temporal pairwise difference
        pi_pred  = self.compute_pairwise_conv(preds, self.pi_diff_ke)
        pi_pred  = torch.clamp(torch.abs(pi_pred), 1e-8, 5.0) #[N, bs*ht*wd]

        pi_target = self.compute_pairwise_conv_nograd(targets, self.pi_diff_ke)
        pi_target = (torch.abs(pi_target)>0.5).float() #[N, bs*ht*wd]

        if weights is not None:
            pi_weight = self.compute_pairwise_conv_nograd(weights, self.pi_sum_ke)

        # if BG, central pixel only considering FG pixels.
        if BG:
            idx = targets.nonzero()
        else:
            idx = torch.cusum(torch.ones([1, bs*ht*wd], dtype=torch.int), axis=1) - 1

        # compute loss
        if len(idx) > 0:
            pi_pred   = pi_pred[:, idx]
            pi_target = pi_target[:, idx]
            if self.loss_type == 'l1':
                loss = pi_l1_loss(pi_pred, pi_target, self.margin, self.pos_wght)
            else: # self.loss_type=='DM-exp':
                loss = pi_exp_loss(pi_pred, pi_target, self.margin, self.pos_wght)

            if weights is not None:
                pi_weight = pi_weight[:, idx]
                loss = torch.sum(torch.mul(loss, pi_weight), dim=-1)

            eval_pi0 = (pi_pred*(pi_target==0).float()).mean()
            eval_pi1 = (pi_pred*(pi_target==1).float()).mean()

            return {'loss':torch.mean(loss),
                    'eval_pi0': eval_pi0,
                    'eval_pi1': eval_pi1}
        else:
            return None
