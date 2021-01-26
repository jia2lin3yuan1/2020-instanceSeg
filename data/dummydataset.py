import os
import os.path as osp
import sys
import cv2
import random
import numpy as np
from scipy import misc as smisc
from glob import glob

import torch
import torch.utils.data as data
import torch.nn.functional as F

from .base_dataset import Detection
from .base_dataset import FromImageAnnotationTransform as AnnotationTransform

class DummyDetection(Detection):
    """`
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (semImg, instImg) and transforms it to bbox+cls.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, mask_out_ch=1, info_file=None, option=None,
                 transform=None, target_transform=None,
                 dataset_name='dummy', running_mode='test', model_mode='InstSeg'):
        '''
        Args:running_mode: 'train' | 'val' | 'test'
             model_mode: 'InstSeg' | 'SemSeg' | 'ObjDet'
        '''
        super(DummyDetection, self).__init__(image_path,
                                            mask_out_ch,
                                            option.sem_weights,
                                            transform,
                                            AnnotationTransform(option),
                                            running_mode,
                                            model_mode)

        self.ignore_label   = option.ignore_label
        self.name           = dataset_name
        self.image_set      = running_mode
        self.ids           = self._load_image_set_index()


    def _load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        base_dir = self.root
        fdir_list = os.listdir(base_dir)

        # get image list from each sub-folder.
        key = '.jpg'
        image_set_index = []
        for fdir in fdir_list:
            glob_imgs = glob(osp.join(base_dir, fdir, '*'+key))
            img_list = [osp.join(fdir, osp.basename(v).split(key)[0]) for v in glob_imgs]
            image_set_index += img_list

        return image_set_index

    def _instanceImg_parsing(self, instI):
        '''
        @func:
        '''
        return instI

    def _semanticImg_parsing(self, semI):
        '''
        @func:
        '''
        return semI

    def save_subpath(self, index, result_path='', subPath=''):
        fname = self.ids[index]
        sub_folder, fname = fname.split('/')
        result_path =  osp.join(result_path, subPath, sub_folder)
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
        fpath = self.ids[index]
        sub_folder, fname = fpath.split('/')

        # read image
        image_file = osp.join(self.root, sub_folder, fname + '.jpg')
        img = cv2.imread(image_file)
        height, width, _ = img.shape

        num_crowds = 0
        if self.has_gt:
            anns = self.pull_anno(index)
            semI, masks, target = anns['sem'], anns['inst_mask'], anns['bbox']
        else:
            semI, masks, targets = None, None, None

        if target is None:
            masks  = np.zeros([1, height, width], dtype=np.float)
            target = np.array([[0,0,1,1,0]])

        # add BG semantic channels, for panoptic segmentation
        if semI is not None:
            sem_bgs =np.asarray([[0,0,1,1,0]]*self.sem_fg_stCH)
            sem_bg_maskI = np.zeros([self.sem_fg_stCH, height, width])
            for k in range(self.sem_fg_stCH):
                sem_bg_maskI[k] = (semI==k).astype(np.float)
            masks  = np.concatenate([sem_bg_maskI, masks], axis=0)
            target = np.concatenate([sem_bgs, target], axis=0)

        # transform for augmentation
        if self.transform is not None:
            img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                                {'num_crowds': num_crowds, 'labels': target[:, 4]})
            # num_crowds is stored inheirted from coco dataset
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
        fpath = self.ids[index]
        sub_folder, fname = fpath.split('/')
        image_file = osp.join(self.root, sub_folder, fname + '.jpg')
        bgrI       =  cv2.imread(image_file)
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
        fpath = self.ids[index]
        sub_folder, fname = fpath.split('/')
        semI, instI = None, None

        # obtain bbox and mask
        if self.target_transform is not None and semI is not None and instI is not None:
            trans_src = [semI, instI]
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

