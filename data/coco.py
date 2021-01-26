import os
import os.path as osp
import sys
import cv2
import numpy as np
import random

import torch
import torch.utils.data as data
import torch.nn.functional as F

from .base_dataset import Detection
from .base_dataset import FromBboxesAnnotationTransform as AnnotationTransform

class COCODetection(Detection):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, mask_out_ch=1, info_file=None, option=None,
                 transform=None, target_transform=None,
                 dataset_name='MS COCO', running_mode='test', model_mode='InstSeg'):
        '''
        Args:running_mode: 'train' | 'val' | 'test'
             model_mode: 'InstSeg' | 'SemSeg' | 'ObjDet'
        '''
        super(COCODetection, self).__init__(image_path,
                                            mask_out_ch,
                                            option.sem_weights,
                                            transform,
                                            AnnotationTransform(option),
                                            running_mode,
                                            model_mode,
                                            option.sem_fg_stCH)

        # Do this here because we have too many things named COCO
        from pycocotools.coco import COCO

        self.coco = COCO(info_file)

        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not self.has_gt:
            self.ids = list(self.coco.imgs.keys())

        self.name = dataset_name
        self.panoptic_seg = False
        self.num_classes = len(option.class_names)


    def save_subpath(self, index, result_path='', subPath=''):
        img_id = self.ids[index]
        fname = self.coco.loadImgs(img_id)[0]['file_name']
        result_path   = osp.join(result_path, sub_path)
        os.makedirs(result_path, exist_ok=True)
        return {'fname': fname,
                'out_dir': result_path,
                'file_key': self.ids[index]}

    def construct_semantic_image(self, inst_masks, targets):
        '''
        Args:
            inst_masks: numpy array, in [N, ht, wd]
            targets: [N, 5], with element (x0, y0, x1, y1, cls_id), -1 for crowed object
        Returns:
        '''
        ht, wd = inst_masks.shape[1:]
        instI = np.concatenate([np.zeros([1, ht, wd]), inst_masks], axis=0).argmax(axis=0)
        rpl_dict = {0:0}
        for k, ele in enumerate(targets):
            sem_id = 0 if ele[-1] < 0 else
                    (ele[-1] if self.panoptic_seg else self.label_map[ele[-1]])
            rpl_dict[k+1] = sem_id

        semI = np.vectorize(rpl_dict.get)(instI)
        return semI

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

        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)

        img = cv2.imread(path)
        height, width, _ = img.shape

        if self.has_gt:
            anns = self.pull_anno(index)
            semI, masks, target = anns['sem'], anns['inst_mask'], anns['bbox']
            num_crowds          = anns['num_crowds']
        else:
            semI, masks, targets = None, None, None

        if not self.has_gt or len(target)==0:
            masks      = np.zeros([1,height, width], dtype=np.float)
            target     = np.array([[0,0,1,1,-1]])
            num_crowds = 0

        # add BG semantic channels, for panoptic segmentation
        if self.has_gt:
            semI = self.construct_semantic_image(masks, target)
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

            # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
            num_crowds = labels['num_crowds']
            labels     = labels['labels']
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        if target.shape[0] == 0:
            print('Warning: Augmentation output an example with no ground truth. Resampling...')
            return self.pull_item(random.randint(0, len(self.ids)-1))

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
        img_id = self.ids[index]
        path   = self.coco.loadImgs(img_id)[0]['file_name']
        bgrI   = cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)
        return {'rgb': bgrI[..., [2,1,0]]}

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
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
        target  = self.coco.loadAnns(ann_ids)

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object.
        # Both during testing and training, consider these crowds as neutral.
        # Ensure sure that all crowd annotations are at the end of the array
        crowd  = [x for x in target if     ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        for x in crowd:
            x['category_id'] = -1
        target += crowd
        num_crowds = len(crowd)

        # obtain bbox and mask
        if self.target_transform is not None and len(target) > 0:
            target = self.target_transform(target, width, height)
            target = np.array(target)

            if len(target) == 0:
                masks, target = None, None
            else:
                # Pool all the masks for this image into one [num_objects,height,width] matrix
                masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
                masks = np.vstack(masks)
                masks = masks.reshape(-1, height, width) # (N, ht, wd)
        else:
            masks, target = None, None

        return {'bbox': target,
                'inst_mask': masks,
                'sem': semI,
                'inst': instI,
                'num_crowds': num_crowds}

