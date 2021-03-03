import os
import os.path as osp
import sys
import cv2
import random
import numpy as np
from scipy import misc as smisc

import torch
import torch.utils.data as data
import torch.nn.functional as F

from .base_dataset import Detection
from .base_dataset import FromImageAnnotationTransform as AnnotationTransform

class PASCALDetection(Detection):
    """`
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of PASCAL images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (semImg, instImg) and transforms it to bbox+cls.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, mask_out_ch=1, info_file=None, option=None,
                 transform=None, target_transform=None,
                 dataset_name='pascal_voc', running_mode='test', model_mode='InstSeg'):
        '''
        Args:running_mode: 'train' | 'val' | 'test'
             model_mode: 'InstSeg' | 'SemSeg' | 'ObjDet'
        '''
        super(PASCALDetection, self).__init__(image_path,
                                            mask_out_ch,
                                            option.sem_weights,
                                            transform,
                                            AnnotationTransform(option),
                                            running_mode,
                                            model_mode)

        self.ignore_label = option.ignore_label
        self.name = dataset_name

        with open(info_file) as f:
            self.ids = [x.strip() for x in f.readlines()]

    def save_subpath(self, index, result_path='', subPath=''):
        fname    = self.ids[index]
        result_path = osp.join(result_path, sub_path)
        os.makedirs(osp.join(result_path), exist_ok=True)
        return {'fname': fname,
                'out_dir': result_path,
                'file_key': self.ids[index]}

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        img_id = self.ids[index]

        # read image
        img = cv2.imread(osp.join(self.root, 'JPEGImages', img_id+'.jpg'))
        height, width, _ = img.shape

        num_crowds = 0
        if self.has_gt:
            anns = self.pull_anno(index)
            semI, masks, target = anns['sem'], anns['inst_mask'], anns['bbox']
        else:
            semI, masks, target = None, None, None

        if target is None:
            masks  = np.zeros([1, height, width], dtype=np.float)
            target = np.array([[0,0,1.,1.,0]])

        # add BG semantic channels, for panoptic segmentation
        if semI is not None:
            sem_bgs         = np.asarray([[0,0,1.,1.,0]])
            sem_bg_maskI    = np.zeros([1, height, width])
            sem_bg_maskI[0] = (semI==0).astype(np.float)
            masks  = np.concatenate([sem_bg_maskI, masks], axis=0)
            target = np.concatenate([sem_bgs, target], axis=0)

        # transform for augmentation
        if self.transform is not None:
            target = np.array(target)
            img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                                         {'num_crowds': num_crowds, 'labels': target[:, 4]})

            # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
            num_crowds = labels['num_crowds']
            labels     = labels['labels']
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, masks, height, width, num_crowds


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            dict of network input
        '''
        fname = self.ids[index]
        bgrI  = cv2.imread(osp.path.join(self.root, 'JPEGImages', fname+'.jpg'), cv2.IMREAD_COLOR)

        return {'rgb': bgrI[:, :, [2,1,0]]}


    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            dict of annotations -- bbox: list of bbox in [x0,y0,x1,y1,clsId]
                                   inst_mask: object mask in array [N, ht, wd]
                                   sem: sem label image in [ht, wd]
                                   inst: inst label image in [ht, wd]
        '''
        fname = self.ids[index]
        semI = smisc.imread(os.path.join(self.root, 'cls', fname+'.png'), mode='P')
        instI = smisc.imread(os.path.join(self.root, 'inst', fname+'.png'), mode='P')
        height, width = semI.shape[:2]

        # obtain bbox and mask
        if self.target_transform is not None:
            trans_src = [semI*(semI!=self.ignore_label), instI*(instI!=self.ignore_label)]
            target = self.target_transform(trans_src, width, height)
            target = np.array(target)
            if len(target) == 0:
                target, masks = None, None
            else:
                # instance binary masks in different channels
                cor_instI = trans_src[1]
                masks = np.eye(cor_instI.max()+1)[cor_instI]
                eff_chs = [0] + [ele for ele in np.unique(cor_instI) if ele > 0]
                masks = masks[..., eff_chs]
                masks = np.transpose(masks[:,:,1:], [2,0,1]).astype(np.float)
        else:
            target, masks = None, None

        return {'bbox': target,
                'inst_mask': masks,
                'sem': semI,
                'inst': instI}

