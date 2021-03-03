import numpy as np
from skimage import measure as smeasure

import torch
import torch.utils.data as data
import torch.nn.functional as F

def get_label_map(cfg):
    if cfg.label_map is None:
        return {x+1: x+1 for x in range(len(cfg.class_names))}
    else:
        return cfg.label_map

def extract_target_from_image(semI, instI, ignore_label=255):
    '''
    @func: extract instances and return as list of [x0,y0,x1,y1,cls_id]
    '''
    props = smeasure.regionprops(instI.astype(np.uint16))

    flag, res = False, []
    for prop in props:
        # find sem category
        y0,x0,y1,x1 = prop.bbox
        coord = prop.coords
        semIds, cnts = np.unique(semI[coord[:,0], coord[:,1]], return_counts=True)
        sem_id = semIds[cnts.argmax()]
        if sem_id not in [0, ignore_label]:
            res.append([x0,y0,x1,y1,sem_id])

    return res

class AnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self, dataset_cfg):
        self.label_map = get_label_map(dataset_cfg)

    def __call__(self, target, width, height):
        """
        Args:
            target: contains annotation information
            height (int): image height
            width (int): im age width
        Returns:
            a list containing lists of bounding boxes [xmin, ymin, xmax, ymax, class_idx]
        """
        raise NotImplementedError
    def getLabelMap(self):
        return self.label_map

class FromImageAnnotationTransform(AnnotationTransform):
    """ Transform labeled instance image and semantic image into a tensor of bbox coords and label index as (x0, y0, x1, y1, cls_id)

    """
    def __init__(self, dataset_cfg):
        super(FromImageAnnotationTransform, self).__init__(dataset_cfg)

    def __call__(self, target, width, height):
        """
        Args:
            target (list): annotations, [semI, instI]
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes [xmin, ymin, xmax, ymax, class_idx]
        """
        scale = np.array([width, height, width, height], dtype=np.float32)
        inst_objs = extract_target_from_image(target[0], target[1])

        res = []
        for ele in inst_objs:
            bbox    = ele[:4]
            sem_id  = ele[4]
            final_box = list(np.array([bbox[0], bbox[1], bbox[2], bbox[3]])/scale)
            final_box.append(sem_id)
            res += [final_box]

        return res


class FromBboxesAnnotationTransform(AnnotationTransform):
    """Transforms a COCO-like annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self, dataset_cfg):
        super(FromBboxesAnnotationTransform, self).__init__(dataset_cfg)

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO-like target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes [xmin, ymin, xmax, ymax, class_idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox   = obj['bbox']
                sem_id = obj['category_id']
                if sem_id < 0: # crowd
                    sem_id = 0 #self.label_map[sem_id] - 1
                sem_id = self.label_map[sem_id]
                final_box = list(np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])/scale)
                final_box.append(sem_id)
                res += [final_box]
            else:
                print("No bbox found for object ", obj)

        return res


class Detection(data.Dataset):
    """`
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of images.
        transform (callable, optional): A function/transform that augments the raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (annotation) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, mask_ch=1, sem_weights=None,
                 transform=None, target_transform=None,
                 running_mode='test', model_mode='InstSeg'):
        '''
        Args: model_mode: 'InstSeg' | 'SemSeg' | 'ObjDet'
        '''
        self.root = image_path
        self.sem_weights = sem_weights

        self.transform = transform
        self.target_transform = target_transform

        self.isTrain = True if running_mode=='train' else False
        self.has_gt = False if running_mode=='test' else True
        self.model_mode = model_mode
        self.mask_out_ch = mask_ch


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        # image in [3, ht, wd], target in list of [x0,y0,x1,y1,cls], masks in [ch, ht, wd]
        im, target, masks, h, w, num_crowds = self.pull_item(index)

        if self.model_mode == 'ObjDet':
            print('ObjDet')
            pass
        else:
            instGT, semGT, wghts, target = self.construct_inst_sem_label(masks, target,
                                                                 self.mask_out_ch,
                                                                 self.sem_weights,
                                                                 num_crowds=num_crowds,
                                                                 isTrain=self.isTrain)
            if self.model_mode == 'InstSeg':
                masks = torch.from_numpy(np.concatenate([instGT, wghts], axis=0))
            else:
                masks= torch.from_numpy(semGT)

        if False:
            from matplotlib import pyplot as plt
            vis_img = im.permute(1,2,0).detach().numpy()
            fig, ax = plt.subplots(1, 4)
            ax[0].imshow(vis_img[:,:,:3])
            ax[1].imshow(instGT.argmax(axis=0))
            ax[2].imshow(semGT[0])
            ax[3].imshow(wghts[0])
            plt.show();
            ta = self.pull_item(index)

        return im, (target, masks, num_crowds)


    def __len__(self):
        return len(self.ids)


    def pull_image(self, index):
        '''
        Read network input.
        For network input, except RGB image, there could also contains motion | depth, then read
            and convert it as RGB format for visualization.
        '''
        pass

    def pull_anno(self, index):
        '''
        Read GT. and parse it into [bboxes, semI, inst_masks], in which,
                | bboxes is 2D array in shape [N, 5], to be [x0,y0,x1,y1,clsId]
                | semI is a 2D mask in shape [ht, wd]
                | inst_masks is a 3D mask, in shape [N, ht, wd]
        '''
        pass

    def save_subpath(self, index, result_path='', subPath=''):
        """
        Parse directory and file-name for saving result,
              file_key for store result in a dictionary
        """
        pass


    def construct_inst_sem_label(self, masks, target, mask_out_ch=64,
                                        sem_weights=None, num_crowds=0,
                                        isTrain=False, fg_thr=0.3):
        def _compute_weights_one_instance(mask, sem_id, base_cnt):
            inst_wght = np.cbrt(base_cnt/(mask.sum()+1.))
            sem_wght  = 1.0 if sem_weights is None else sem_weights[sem_id]
            return np.clip(sem_wght*inst_wght, 1.0, 10.0)

        # main process.
        bgI, masks = masks[0], masks[1:]
        target = target[1:]

        # Remove crowed objects in training. designed for coco
        if num_crowds > 0:
            masks, target = masks[:-num_crowds], target[:-num_crowds]

        # if there are more objects than expected, deal with large object first
        num_obj, ht, wd = masks.shape
        if num_obj>1:
            obj_areas = np.reshape(masks, [num_obj, -1]).sum(axis=-1)
            idx = sorted(range(num_obj), key=lambda i: -obj_areas[i])
        else:
            idx = np.arange(num_obj)

        # allocate buffer
        instGT = np.zeros([mask_out_ch, ht, wd], dtype=np.float32)
        semGT  = np.zeros([1, ht, wd], dtype=np.float32)
        wghts  = np.zeros_like(semGT, dtype=np.float32)
        new_target = np.zeros([mask_out_ch, 5], dtype=np.float32)
        base_cnt = float(ht*wd)

        # compute background value
        semGT[0, bgI>0]  = 0
        instGT[0, bgI>0] = 1
        wghts[0, bgI>0]  = _compute_weights_one_instance(instGT[0], 0, base_cnt)

        # compute FG value
        for ik, k in enumerate(idx[:mask_out_ch-1]):
            new_target[ik+1] = target[k]

            # FG object has higher prior than BG stuff
            # smaller object has higher priority to cover large object
            instGT[:, masks[k]>fg_thr] = 0
            instGT[ik+1]     = masks[k]>fg_thr

            semGT[0, masks[k]>0]  = target[k, -1]
            wghts[0, masks[k]>0]  = _compute_weights_one_instance(masks[k]>0,
                                                                 int(target[k, -1]),
                                                                 base_cnt)
        return instGT, semGT, wghts, new_target


    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        raise NotImplementedError

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        raise NotImplementedError

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        raise NotImplementedError

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def enforce_size(img, targets, masks, num_crowds, new_w, new_h):
    """ Ensures that the image is the given size without distorting aspect ratio. """
    with torch.no_grad():
        _, h, w = img.size()

        if False: #h == new_h and w == new_w:
            return img, targets, masks, num_crowds

        # Resize the image so that it fits within new_w, new_h
        w_prime = new_w
        h_prime = h * new_w / w

        if h_prime > new_h:
            w_prime *= new_h / h_prime
            h_prime = new_h

        w_prime = int(w_prime)
        h_prime = int(h_prime)

        # Do all the resizing
        img = F.interpolate(img.unsqueeze(0),
                            (h_prime, w_prime),
                            mode='bilinear',
                            align_corners=False)
        img.squeeze_(0)

        # Act like each object is a color channel
        masks = F.interpolate(masks.unsqueeze(0),
                              (h_prime, w_prime),
                              mode='nearest')
        masks.squeeze_(0)

        # Scale bounding boxes (this will put them in the top left corner in the case of padding)
        targets[:, [0, 2]] *= (w_prime / new_w)
        targets[:, [1, 3]] *= (h_prime / new_h)

        # Finally, pad everything to be the new_w, new_h
        pad_dims = (0, new_w - w_prime, 0, new_h - h_prime)
        img   = F.pad(  img, pad_dims, mode='constant', value=0)
        masks = F.pad(masks, pad_dims, mode='constant', value=0)

        return img, targets, masks, num_crowds




def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and (lists of annotations, masks)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
                on 0 dim. The output gt is a tuple of annotations and masks.
    """
    targets = []
    imgs = []
    masks = []
    num_crowds = []

    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1][0]))
        masks.append(torch.FloatTensor(sample[1][1]))
        num_crowds.append(sample[1][2])

    return imgs, (targets, masks, num_crowds)


