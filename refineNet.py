import numpy as np
from torchvision.ops import roi_align
import torch
import torch.nn as nn
import torch.nn.functional as F

class RoIAlign(nn.Module):
    def __init__(self, out_size=(14, 14), aligned=True):
        super(RoIAlign, self).__init__()
        self.out_size = out_size
        self.aligned = aligned

    def forward(self, input, bboxes, spatial_scale=1.0):
        '''
        @Param: input--tensor of size [bs, ch, ht, wd]
                bboxes -- float tensor of size [N,,,, 5] with [b_idx, x0, y0, x1, y1], or list of
                        tensor with size [n_i, 4], with each one corresponding to one batch example
                spatial_scale -- scale that map feature coordinates to bboxes' coordinates.
                                 ( e.x. feature (8, 8) map to box (16,16) with scale 0.5)
        @Output: tensor of size [N, ch, out_size[0], out_size[1]]

        '''
        rois = roi_align(input.float(), bboxes.float(), spatial_scale=spatial_scale,
                         output_size=self.out_size) # , aligned=self.aligned)
        '''
        current torch version is 1.3.0, its roi_align has no argument aligned
        '''
        return rois


class RoIExtractor(nn.Module):
    def __init__(self, roi_size = (14, 14), fea_channels=[256], doConv=True):
        super(RoIExtractor, self).__init__()

        self.roi_align  = RoIAlign(out_size=roi_size)
        relu_func = nn.LeakyReLU(0.1)

        self.doConv = doConv
        if doConv:
            blocks = []
            for in_channel in fea_channels:
                one_block = nn.Sequential(
                            nn.Conv2d(in_channel, 256, kernel_size=3, padding=(1,1), padding_mode='replicate'),
                            nn.BatchNorm2d(256),
                            #relu_func,
                        )
                blocks.append(one_block)
            self.conv_layers = nn.ModuleList(blocks)

    def forward(self, bboxes, features, fea_scales=[1.0]):
        '''
        @Param: bboxes -- float tensor of size [N,,,, 5] with [b_idx, x0, y0, x1, y1], or list of
                        tensor with size [n_i, 4], with each one corresponding to one batch example
                features -- list of features to be extracted, in size [bs, ch, ht', wd']
                scales -- list of scale ratio that map feature coordinates to bboxes' coordinates

        @ Output: list of features extract from different layer
        '''
        ret = []
        for k in range(len(features)):
            x = self.roi_align(features[k], bboxes, spatial_scale=fea_scales[k])
            if self.doConv:
                x = self.conv_layers[k](x)
            ret.append(x)
        return ret


class RefineNet(nn.Module):
    '''
    this class takes (backbone features, predicted instance label) and object candidates,
    it performs roi_align to extract features, then predict class score, iou score, refinement
    mask for each candidate
    '''
    def __init__(self, roi_size=(14, 14), fea_layers=[256],
                        num_classes=21, pi_margin=1.0):
        super(RefineNet, self).__init__()
        self.roi_size    = roi_size
        self.num_classes = num_classes
        self.pi_margin   = pi_margin

        self.RoI_extractor_mask = RoIExtractor(roi_size = roi_size, doConv=False)
        self.RoI_extractor_net  = RoIExtractor(roi_size = roi_size, doConv=True)
        relu_func = nn.LeakyReLU(0.1)

        in_channels = sum(fea_layers) + 1
        hiddens = [256, 512]
        self.conv_0 = nn.Sequential(
                      nn.Conv2d(in_channels, hiddens[0], kernel_size=1),
                      relu_func,
                      nn.Conv2d(hiddens[0], hiddens[0], kernel_size=3, padding=(1,1), padding_mode='replicate'),
                      relu_func,
                      nn.Conv2d(hiddens[0], hiddens[0], kernel_size=3, padding=(1,1), padding_mode='replicate'),
                      relu_func,

                      nn.Conv2d(hiddens[0], hiddens[1], kernel_size=3, padding=(1,1), padding_mode='replicate'),
                      relu_func,
                      nn.Conv2d(hiddens[1], hiddens[1], kernel_size=3, padding=(1,1), padding_mode='replicate'),
                      #nn.BatchNorm2d(hiddens[1]),
                    )

        cls_dim = (roi_size[0]//2)*(roi_size[1]//2)*hiddens[1]
        self.cls_iou_linear = nn.Sequential(
                                 relu_func,
                                 nn.Linear(cls_dim, 1024),
                                 relu_func,
                          )
        self.cls_linear_logits = nn.Sequential(
                                 nn.Linear(1024, 512),
                                 relu_func,
                                 nn.Linear(512, self.num_classes)
                          )
        self.iou_linear_logits = nn.Sequential(
                                 nn.Linear(1024, 512),
                                 relu_func,
                                 nn.Linear(512, 1),
                                 nn.ReLU()
                          )
        # mask
        self.conv_mask = nn.Sequential(
                       relu_func,
                       nn.Conv2d(hiddens[1], hiddens[0], kernel_size=1, padding_mode='replicate'),
                       nn.UpsamplingBilinear2d(size=(2*roi_size[0], 2*roi_size[1])),
                       relu_func,
                       nn.Conv2d(hiddens[0], hiddens[0], kernel_size=3, padding=(1,1), padding_mode='replicate'),
                       relu_func,
                       nn.Conv2d(hiddens[0], hiddens[0], kernel_size=3, padding=(1,1), padding_mode='replicate'),

                       #nn.BatchNorm2d(hiddens[0]),
                       relu_func
                    )
        self.mask_fcn_logits = nn.Conv2d(hiddens[0], 1, kernel_size=1)

    def vis_debug_roi_align(self, full_feature, bbox, roi_out, scale=1.0):
        '''
        bbox: [bk, x0,y0,x1,y1], used to sort so to vis large roi first
        '''
        from matplotlib import pyplot as plt
        roi  = roi_out.cpu().detach().numpy()
        full = full_feature.cpu().detach().numpy()

        fig, ax  = plt.subplots(4,2)
        ax[0,0].imshow(full[0])
        ax[0,1].imshow(full[1])

        bbox_np = (bbox).int().cpu().detach().numpy()
        bbox_np[:, 1:5] = bbox_np[:, 1:5]*scale
        bbox_wd = bbox_np[:,3] - bbox_np[:,1]
        idx = sorted(range(len(bbox_wd)), key=lambda i:-bbox_wd[i])
        for i, k in enumerate(idx[:4]):
            cand = bbox_np[k]
            print(cand)
            bk, x0, y0, x1, y1 = cand[0], int(cand[1]), int(cand[2]), int(cand[3]), int(cand[4])
            ax[i+1,0].imshow(roi[k])
            ax[i+1,1].imshow(full[bk][y0:y1+1, x0:x1+1])
        plt.show()
        import pdb; pdb.set_trace()


    def forward(self, in_masks, net_fea, rois, vis_debug=False):
        '''
        @Param: in_masks -- real instance label prediction from proto-branch,
                        for SC-arch, size [bs, 1, ht, wd]
                net_fea -- list of backbone network features, in size [bs, ch, ht', wd']
                rois -- tensor of bboxes information, size [N, x], with,  coords in in_masks
                        resolution. ::
                        for SC-arch, [bs_idx, x0, y0, x1, y1, integer_label, real_label],
        @Out: a dict includes, 'cls' -- [N, num_class]
                               'iou' -- [N, 1]
                               'seg' -- [N, 1, h, w]
        '''
        bs, ch, ht, wd = in_masks.size()
        roi_bboxes = rois[:, :5]

        # features
        with torch.no_grad():
            mask_feas = self.RoI_extractor_mask(roi_bboxes, [in_masks], fea_scales=[1.0])[0]
            mask_feas = torch.abs(mask_feas - rois[:,-1][:, None, None, None])
            mask_feas = torch.clamp(self.pi_margin - mask_feas, min=0)

        # debug rois
        if vis_debug:
            labelImgs = in_masks[:,0] if ch==1 else in_masks.argmax(axis=1)
            self.vis_debug_roi_align(labelImgs, roi_bboxes, mask_feas[:,0], scale=1.0)

        if isinstance(net_fea, list):
            fea_scales = [ele.size(2)/in_masks.size(2) for ele in net_fea]
            backbone_feas = self.RoI_extractor_net(roi_bboxes, net_fea, fea_scales=fea_scales)
        else:
            fea_scales = [net_fea.size(2)/in_masks.size(2)]
            backbone_feas = self.RoI_extractor_net(roi_bboxes, [net_fea], fea_scales=fea_scales)

        # network
        feature = torch.cat(backbone_feas + [mask_feas], dim=1)
        fea_0 = self.conv_0(feature)

        cls_iou_fea = nn.MaxPool2d(2, stride=2)(fea_0)
        cls_iou_fea = self.cls_iou_linear(cls_iou_fea.view(fea_0.size(0), -1))
        cls_logits  = self.cls_linear_logits(cls_iou_fea)
        iou_logits  = self.iou_linear_logits(cls_iou_fea)

        mask_layer = self.conv_mask(fea_0)
        mask_logits = self.mask_fcn_logits(mask_layer)

        return {'cls': cls_logits,
                'iou': iou_logits,
                'mask':mask_logits}

