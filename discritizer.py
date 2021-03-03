import numpy as np
from skimage import measure as skmeasure

import torch
from torch import nn
from torch.nn import functional as F


def vis_meanshift_result(oriI, mf_labelI, new_labelM):
    '''
    @Param: oriI | mf_labelI -- [ht, wd]
            new_labelM -- [N, ht, wd] for N objects
    '''
    new_labelI = new_labelM.argmax(axis=0) + 1
    new_labelI[new_labelM.sum(axis=0)==0] = 0

    from matplotlib import pyplot as plt
    fig, ax =plt.subplots(1,3)
    ax[0].imshow(oriI)
    ax[1].imshow(mf_labelI)
    ax[2].imshow(new_labelI)

    _ = ax[0].set_title('real label')
    _ = ax[1].set_title('meanshift result')
    _ = ax[2].set_title('final candidates')

    plt.show()


def extract_candidates_np(labelI, intensityI, bidx, cand_params):
     #bg_thr=0.1, size_thr=3):
    '''
    @Func: extract each connected area as bboxes from labelI. With no merge of un-connected rois.
    @Param: labelI -- numpy array, [ht, wd]
            intensityI -- numpy array, [ht, wd]
            bidx -- idx of the image in batch
            cand_params -- parameters for remove false proposals
    @Output: updated onehot label tensor, in size [N+1, ht, wd]
             bboxes tensor, in size [N, 7]
    '''
    new_labelI = skmeasure.label(labelI)
    props      = skmeasure.regionprops(new_labelI, intensity_image=intensityI)
    if cand_params is not None:
        bg_thr     = cand_params['bg_thr']
        size_thr   = cand_params['size_thr']
        num_keep   = cand_params['num_keep']
    else:
        bg_thr, size_thr, num_keep = 0.2, 5, 100

    labelI_onehot, bboxes = [], []
    sort_idxes = sorted([k for k in range(len(props))], key=lambda i: -props[i].area)
    for k in sort_idxes:
        prop = props[k]
        rm_det = True if prop.mean_intensity<bg_thr or prop.area<size_thr else False
        if cand_params is not None and 'a2p_ratio_thr' in cand_params:
            a2p_ratio_thr = cand_params['a2p_ratio_thr']
            a2p_pm_thr    = cand_params['a2p_pm_thr']
            if prop.perimeter > a2p_pm_thr and  prop.area/prop.perimeter < a2p_ratio_thr:
                rm_det = True

        if not rm_det and len(bboxes) < num_keep:
            x0,y0,x1,y1 = prop.bbox
            integer_label = len(labelI_onehot)
            bboxes.append([bidx, x0, y0, x1-1,y1-1, integer_label, prop.mean_intensity])
            labelI_onehot.append(new_labelI==prop.label)

    # if there is no objects in the image, send the full image for further processing
    if len(bboxes) == 0:
        bboxes = [[bidx, 0, 0, labelI.shape[0]-1, labelI.shape[1]-1, 0, 0.0]]
        labelI_onehot = [new_labelI*0]

    # conver to torch tensor
    labelI_onehot = torch.as_tensor(np.stack(labelI_onehot, axis=0)).float() # [N, ht, wd]
    bboxes     = torch.as_tensor(bboxes).float()

    return labelI_onehot, bboxes


def extract_candidates(labelImgs, intensityImgs, xAxisI, yAxisI, ht, wd,
                           cand_params=None, merge_thr=0.1):
    '''
    @Func: extract each connnected components and merge components with similar intensity as RoIs.
    @Param: labelImgs -- tensor in size [bs, ht*wd]
            intensityImgs -- tensor in size [bs, 1, ht*wd]
            xAxisI -- tensor in size [1, ht*wd], value is x axis
            yAxisI -- tensor in size [1, ht*wd], value is y axis
            cand_params -- parameters for remove false proposals,
            merge_thr -- intensity threshold for merging non-connected components
    @Output: updated labelI and list of bboxes, each in size [N, 7]
    '''


    ## todo::yjl:: check logic, axis_x why 0 elements
    def empty_case():
        # no candidates to process, construct a all-0 BG case
        # [bth_idx, x0, y0, x1, y1, int_label, mean_intensity]
        if labelImgs.is_cuda:
            cur_bboxes = torch.FloatTensor([[bk, 0, 0, 2, 2, 0, 0.0]]).cuda()
        else:
            cur_bboxes = torch.FloatTensor([[bk, 0, 0, 2, 2, 0, 0.0]])

        img = labelImgs[0].reshape(1, ht, wd)*0.
        return cur_bboxes, img

    def merge_disconnected_regions(mean_intensity, FG_onehot):
        # mean_intensity -- [N']
        # FG_onehot -- [N', ht*wd]
        diff  = torch.abs(mean_intensity[:, None] - mean_intensity[None, :]) #[N', N']
        merge = diff < merge_thr
        merge_idxes, new_masks = [], []
        for i in range(diff.size(0)):
            if merge[i].sum() == 1:
                continue

            # flag to remove duplicate merging
            flag = True
            if len(merge_idxes) > 0:
                siml = torch.logical_xor(merge[i][None, :], merge[merge_idxes]).sum(axis=-1)
                flag = False if 0 in siml else True
            if flag:
                merge_idxes.append(i)
                new_mask = (FG_onehot[merge[i]].sum(axis=0)>0).float() #[ht*wd]
                new_masks.append(new_mask[None, :])
        return new_masks

    ## ## #  Main process
    if cand_params is not None:
        bg_thr     = cand_params['bg_thr']
        size_thr   = cand_params['size_thr']
        num_keep   = cand_params['num_keep']
    else:
        bg_thr, size_thr, num_keep = 0.2, 5, 100

    bs = labelImgs.size(0)
    new_labelI, bboxes = [], []
    for bk in range(bs):
        labelI          = labelImgs[bk] #[ht*wd]
        intensityI      = intensityImgs[bk] # [1, ht*wd]
        N               = labelI.max() + 1
        labelI_onehot_0 = torch.eye(N)[labelI.long()].permute(1, 0) #[N, ht*wd]

        # remove BG masks
        area           = labelI_onehot_0.sum(axis=-1) # [N]
        mean_intensity = (labelI_onehot_0 * intensityI).sum(axis=-1)/(area+1) #[N]
        keep           = mean_intensity > bg_thr    # bool tensor in size [N]

        # if no possible candidates, skip
        if not keep.max():
            cur_bboxes, labelI_onehot = empty_case()
        else:
            if keep.sum()>num_keep:
                # select the top K of area for further process
                area[torch.logical_not(keep)] = 0
                idxes = torch.topk(area, num_keep)[1]
                keep[:] = False
                keep[idxes] = True

            # get each connected components and their merging result
            FG_onehot      = labelI_onehot_0[keep] # [N', ht*wd]
            mean_intensity = mean_intensity[keep]  #[N']
            new_masks      = merge_disconnected_regions(mean_intensity, FG_onehot)
            labelI_onehot  = torch.cat([FG_onehot] + new_masks, axis=0) #[N*, ht*wd]

            # properties
            area           = labelI_onehot.sum(axis=-1) # [N*]
            mean_intensity = (labelI_onehot * intensityI).sum(axis=-1)/(area+1) #[N*]

            # remove noise candidates
            keep           = area > size_thr
            if not keep.max():
                cur_bboxes, labelI_onehot = empty_case()
            else:
                labelI_onehot  = labelI_onehot[keep]   #[N**, ht*wd]
                mean_intensity = mean_intensity[keep]  #[N**]

                base_tmpl      = torch.ones_like(mean_intensity, dtype=torch.float)
                bth_idx        = base_tmpl * bk  # [N**]
                int_label      = torch.cumsum(base_tmpl, axis=0)-1 # [N**]

                # bbox coords
                x_axis = labelI_onehot * xAxisI
                y_axis = labelI_onehot * yAxisI
                x1     = x_axis.max(axis=-1)[0] # [N**]
                y1     = y_axis.max(axis=-1)[0] # [N**]

                x_axis[labelI_onehot==0] = wd
                y_axis[labelI_onehot==0] = ht
                x0     = x_axis.min(axis=-1)[0] # [N**]
                y0     = y_axis.min(axis=-1)[0] # [N**]

                # a candidate
                tmp = [bth_idx, x0, y0, x1, y1, int_label, mean_intensity]
                cur_bboxes = torch.stack(tmp, axis=1)
                labelI_onehot = labelI_onehot.view(-1, ht, wd)

        bboxes.append(cur_bboxes)
        new_labelI.append(labelI_onehot)

    return new_labelI, bboxes


def extend_bboxes(bboxes, ht, wd, scale=0.2,
                      max_ext=[16, 16], minbox=[8,8], keep_hw_ratio=False):
    '''
    @func: adjust the crop box, make sure it has certain size.
    @param: boxes -- tensor of size [N, 4], as [x0, y0, x1, y1]
    @output: box -- tensor of size [N, 4]
    '''
    x0, y0, x1, y1 = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]
    cx, cy   = (x0+x1)/2, (y0+y1)/2
    bht, bwd = y1-y0+1, x1-x0+1
    bht = torch.clamp(bht+torch.clamp(bht*scale, max=max_ext[0]), min=minbox[0], max=ht)
    bwd = torch.clamp(bwd+torch.clamp(bwd*scale, max=max_ext[1]), min=minbox[1], max=wd)

    if keep_hw_ratio:
        pht = torch.max(bwd*ht/wd, bht)
        pwd = torch.max(bht*wd/ht, bwd)
    else:
        pht, pwd = bht, bwd

    ext_x0 = torch.clamp(cx-pwd/2, min=0)
    ext_x0 = torch.min(ext_x0, wd-pwd)
    ext_x1 = ext_x0 + pwd-1

    ext_y0 = torch.clamp(cy-pht/2, min=0)
    ext_y0 = torch.min(ext_y0, ht-pht)
    ext_y1 = ext_y0 + pht-1

    ext_bboxes = torch.stack([ext_x0.int(), ext_y0.int(), ext_x1.int(), ext_y1.int()], dim=1)

    return ext_bboxes


class MeanshiftCluster_0(nn.Module):
    '''Adopt the MeanShift algorithm (sklearn.cluster) to group the predicted real instance map
    into segments
    '''
    def __init__(self, bandwidth=0.5):
        super(MeanshiftCluster_0, self).__init__()

        from sklearn.cluster import MeanShift
        self.cluster = MeanShift(bandwidth=bandwidth)

    @torch.no_grad()
    def forward(self, features, cand_params=None):
        '''
        Params:
            features: tensor in size [bs, 1, ht, wd].

        Outputs:
            list of onehot label, each in size [N, ht, wd]
            list of bboxes, each in size [N, 7]
        '''
        bs, _, ht, wd = features.size()
        feaVec = features.view(bs, -1, 1).cpu()
        labels, bboxes = [], []
        for b in range(bs):
            groups    = self.cluster.fit(feaVec[b])
            mf_labelI = groups.labels_ + 1

            # extract bboxes and remove BG regions
            new_labelM, mf_bboxes = extract_candidates_np(mf_labelI, feaImgs[b], b, cand_params)
            labels.append(new_labelM)
            bboxes.append(mf_bboxes)

            if False:
                print('batch element 0: ', mf_bboxes.size(0))
                vis_meanshift_result(features[b, 0].cpu().detach.numpy(), mf_labelI,
                                     new_labelM.cpu().detach().numpy())

        return {'mask': labels,
                'bboxes': bboxes}


class MeanshiftCluster_1(nn.Module):
    '''this class call the meanshift from pymeanshift (https://github.com/fjean/pymeanshift.git) to
    perform the meanshift segmentation part.
    '''
    def __init__(self, spatial_radius=9, range_radius=0.5, min_density=5):
        super(MeanshiftCluster_1, self).__init__()

        from pymeanshift import segment as mf_segment
        self.mf_segment = mf_segment

        self.spatial_radius = spatial_radius
        self.range_radius   = range_radius
        self.min_density    = min_density

    @torch.no_grad()
    def forward(self, features, cand_params=None):
        '''
        Params:
            features: tensor in size [bs, 1, ht, wd].

        Outputs:
            list of onehot label, each in size [N, ht, wd]
            list of bboxes, each in size [N, 7], represents
                    [bs_idx, x0,y0,x1,y1,integer_label, real_label]
        '''
        # move to cpu to run meanshift and extract each bboxes.
        feaImgs = features[:, 0, :, :].cpu().detach().numpy()

        bs, _, ht, wd  = features.size()
        labels, bboxes = [], []
        for b in range(bs):
            scale    = 256./(feaImgs[b].max() + 1.)
            norm_fea = (scale*feaImgs[b]).int()
            _, mf_labelI, _ = self.mf_segment(norm_fea,
                                              self.spatial_radius,
                                              scale*self.range_radius,
                                              self.min_density)
            mf_labelI +=  1

            # extract bboxes and remove BG regions
            new_labelM, mf_bboxes = extract_candidates_np(mf_labelI, feaImgs[b], b, cand_params)
            labels.append(new_labelM)
            bboxes.append(mf_bboxes)

            if False:
                print('batch element 0: ', mf_bboxes.size(0))
                vis_meanshift_result(feaImgs[b], mf_labelI,
                                     new_labelM.cpu().detach().numpy())

        return {'mask': labels,
                'bboxes': bboxes}


class MeanshiftCluster_2(nn.Module):
    '''
    GPU meanshift and labeling, implemented by yjl, at 20201106
    '''
    def __init__(self, spatial_radius=9, range_radius=0.5,
                       epsilon=1e-2, max_iteration=3, cuda=True):
        super(MeanshiftCluster_2, self).__init__()
        self.sradius     = spatial_radius
        self.rradius     = range_radius
        self.epsilon     = epsilon
        self.max_iter    = max_iteration
        self.cuda        = cuda


        ''' here, computed sigma is tested method, for center weights is 2 to 3 times over edge weights
        '''
        self.ssigma     = np.sqrt(2*self.sradius**2)/1.5
        ''' linear mapping rng_wght, so that weight(diff>rradius) has tiny value
        '''
        self.pi      = 3.141592653589793
        self.rng_thr = np.exp(-0.5)/(self.rradius*np.sqrt(2*self.pi))

        self.sdiameter   = spatial_radius *2 + 1
        self.nei_kernel, self.rng_kernel, self.spt_wght = self.create_mf_kernels()


    def gaussian(self, x, sigma=1.0):
        return torch.exp(-0.5*((x/sigma))**2) / (sigma*np.sqrt(2*self.pi))

    def create_mf_kernels(self):
        '''
        @Output: two conv kernel ([out, in ,kh, kw]) to compute neighbour info.
                 spatial gaussian weights (1, out, 1, 1) for each neighbour
        '''
        axis_x, axis_y = np.meshgrid(range(self.sdiameter), range(self.sdiameter))
        cy, cx         = self.sradius, self.sradius
        spt_size   = self.sdiameter*self.sdiameter

        spt_kernel     = torch.sqrt(torch.FloatTensor((axis_x-cx)**2 + (axis_y-cy)**2))
        idxM           = torch.FloatTensor(axis_x + axis_y*self.sdiameter)
        if self.cuda:
            idxM, spt_kernel = idxM.cuda(), spt_kernel.cuda()

        # range_kernel for conv to compute rng_dist
        nei_kernel = torch.eye(spt_size)[idxM.long()].permute(2,0,1) #[K*K, K, K]
        rng_kernel = torch.eye(spt_size)[idxM.long()].permute(2,0,1) #[K*K, K, K]
        rng_kernel[rng_kernel>0] = -1
        rng_kernel[:, cy, cx]   += 1

        # pre-computed sptial weights
        spt_kernel = spt_kernel.reshape(-1) #[K*K]
        spt_wght   = self.gaussian(spt_kernel, sigma=self.ssigma)

        # extend dimension
        nei_kernel, rng_kernel = nei_kernel[:, None, :, :], rng_kernel[:, None, :, :]
        spt_wght = spt_wght[None, :, None, None]

        return nei_kernel, rng_kernel, spt_wght


    def compute_pairwise_conv(self, tensor, kernel):
        """
        @func: compute pairwise relationship
        @param: tensor -- size [bs, 1, ht, wd]
                kernel -- size [N, 1, kht, kwd]

        @output: result in size [bs, N, ht, wd]
        """
        kht, kwd       = kernel.shape[2], kernel.shape[3]
        pad_ht, pad_wd = kht//2, kwd//2
        padT = torch.nn.ReplicationPad2d([pad_wd, pad_wd, pad_ht, pad_ht])(tensor)
        out  = F.conv2d(padT, kernel)  # [bs, N, ht, wd]
        assert(out.shape[2:] == tensor.shape[2:])
        return out

    def meanshift(self, features, has_spatial=True):
        '''
        @func: meanshift segmentation algorithm:
             https://www.cnblogs.com/ariel-dreamland/p/9419154.html#:~:text=Mean%20Shift%E7%AE%97%E6%B3%95%EF%BC%8C%E4%B8%80%E8%88%AC%E6%98%AF,%E7%9B%AE%E6%A0%87%E8%B7%9F%E8%B8%AA%E5%92%8C%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%E3%80%82
        @Param: input - [bs, 1, ht, wd]
        '''
        x = features
        for _ in range(self.max_iter):
            nei_x    = self.compute_pairwise_conv(x, self.nei_kernel)
            rng_diff = self.compute_pairwise_conv(x, self.rng_kernel)
            rng_wght = self.gaussian(rng_diff, self.rradius) # [bs, N, ht, wd]
            flag     = rng_wght<self.rng_thr
            rng_wght[flag] = rng_wght[flag]/10.
            if has_spatial:
                rng_wght = rng_wght * self.spt_wght

            new_x   = ((rng_wght*nei_x).sum(axis=1)/rng_wght.sum(axis=1))[:, None, :, :]
            if torch.abs(new_x-x).max()<self.epsilon:
                return new_x
            else:
                x = new_x
        return x

    @torch.no_grad()
    def forward(self, features, cand_params=None, do_label=True):
        '''
        Params:
            features: tensor in size [bs, 1, ht, wd].

        Outputs:
            list of onehot label, each in size [N, ht, wd]
            list of bboxes, each in size [N, 7], represents
                    [bs_idx, x0,y0,x1,y1,integer_label, real_label]
        '''
        bs, _, ht, wd  = features.size()

        # meanshift smoothing
        relu_feas = nn.ReLU()(features)
        mf_fea = self.meanshift(relu_feas)
        if not do_label:
            return {'mf_fea': mf_fea}
        else:
            mf_fea = mf_fea[:, 0, :, :]

        # discritized label
        labelImgs = torch.round(mf_fea/(self.rradius+1e-2)).int()

        if False:  # CPU running
            labelImgs = labelImgs.cpu()
            feaImgs = relu_feas[:, 0, :, :].cpu().detach().numpy()
            labels, bboxes = [], []
            for b in range(bs):
                mf_labelI = skmeasure.label(labelImgs[b])

                # extract bboxes and remove BG regions
                new_labelM, mf_bboxes = extract_candidates_np(mf_labelI, feaImgs[b], b, cand_params)
                labels.append(new_labelM)
                bboxes.append(mf_bboxes)
        else:
            axis_x, axis_y = np.meshgrid(range(wd), range(ht))
            axis_x = torch.FloatTensor(axis_x).view(1, -1)
            axis_y = torch.FloatTensor(axis_y).view(1, -1)
            if relu_feas.is_cuda:
                axis_x, axis_y = axis_x.cuda(), axis_y.cuda()
            labels, bboxes = extract_candidates(labelImgs.view(bs, -1),
                                                relu_feas.view(bs, 1, -1),
                                                axis_x, axis_y,
                                                ht, wd,
                                                cand_params=cand_params,
                                                merge_thr=self.rradius)
        if False:
            bk = 0
            print('batch element ', bk, ': ', bboxes[bk].size(0))
            vis_meanshift_result(relu_feas[bk, 0].cpu().detach().numpy(),
                                 labelImgs[bk].cpu().detach().numpy(),
                                 labels[bk].cpu().detach().numpy())

        return {'mask': labels,
                'bboxes': bboxes}


class Discritizer(nn.Module):
    ''' This class discritize the real instance label into indexed label,
    '''

    def __init__(self, cand_parameters, bg_thr=0.3, a2p_ratio_thr=1.8, a2p_pm_thr=15):
        super(Discritizer, self).__init__()
        #self.segmentor = MeanshiftCluster_0()
        #self.segmentor = MeanshiftCluster_1()

        spatial_radius = cand_parameters['mf_sradius']
        range_radius   = cand_parameters['mf_rradius']
        self.segmentor = MeanshiftCluster_2(spatial_radius, range_radius)

        self.cand_params = {'bg_thr': bg_thr,
                            'a2p_ratio_thr': a2p_ratio_thr,
                            'a2p_pm_thr': a2p_pm_thr,
                            'num_keep': cand_parameters['mf_num_keep'],
                            'size_thr': cand_parameters['mf_size_thr']}

    @torch.no_grad()
    def forward(self, preds, targets=None):
        '''
        @Params:
            preds -- prediction of the network, in size [bs, 1, ht, wd]
            targets -- ground truth, in size [bs, 1, ht, wd]
        @Output:
            mask -- list of discritized label one hot image, element in size [N, ht, wd]
            bboxes -- list of bboxes for objects, in size [N, 7],
                        as (bs_idx, x0, y0, x1, y1, integer_label, real_label)
        '''
        segments = self.segmentor(preds, self.cand_params)

        # extend bboxes
        bboxes         = torch.cat(segments['bboxes'], axis=0)
        bboxes[:, 1:5] = extend_bboxes(bboxes[:, 1:5], preds.size(2), preds.size(3))

        return {'mask': segments['mask'],
                'bboxes': bboxes}

