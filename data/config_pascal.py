from data.base_config import *


PASCAL_CLASSES_BG = ['bg'] # 0
PASCAL_CLASSES_FG = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', # 1~5
                  'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', # 6~11
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', # 12 ~16
                  'sheep', 'sofa', 'train',  'tvmonitor'] # 17~20
PASCAL_CLASSES = PASCAL_CLASSES_BG + PASCAL_CLASSES_FG

PASCAL_LABEL_FG = {1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16,
                   17: 17, 18: 18, 19: 19, 20: 20}
PASCAL_LABEL_MAP = { 0:0, 255:255}
PASCAL_LABEL_MAP.update(PASCAL_LABEL_FG)

PASCAL_SEM_WEIGHTS = { 0:1.0000,  1:1.0549,  2:1.1200,  3:0.9519,  4:1.0380,
                     5:0.9103,  6:1.1860,  7:0.7786,  8:0.9230,  9:0.7324,
                    10:1.1773, 11:1.1414, 12:0.8612, 13:1.1056, 14:1.1102,
                    15:0.4762, 16:0.9931, 17:1.0510, 18:1.1787, 19:1.1504,
                    20:1.0608, 255:0.0}

# ----------------------- DATASETS ----------------------- #

pascal_dataset_base = Config({
    'name': 'PascalVOC Dataset',

    # Training images and annotations
    'train_images': 'path/to/train/images',
    'train_info':   'path/to/annotation/file',

    # Validation images and annotations.
    'valid_images': 'path/to/valid/images',
    'valid_info':   'path/to/annotation/file',

    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'has_gt': True,

    # A list of names for each of you classes.
    'class_names': PASCAL_CLASSES,
    'ignore_label': 255,

    'label_map': None,
})

pascal2012_dataset = pascal_dataset_base.copy({
    'name': 'PascalVOC 2012',

    'train_images': '../dataset/pascal/',
    'train_info': '../dataset/pascal/train.txt',

    'valid_images': '../dataset/pascal/',
    'valid_info': '../dataset/pascal/val.txt',

    'label_map': PASCAL_LABEL_MAP,
    'sem_weights': PASCAL_SEM_WEIGHTS
})


# ----------------------- CONFIG DEFAULTS ----------------------- #

pascal_base_config = Config({
    'dataset': pascal2012_dataset,
    'num_classes': len(PASCAL_CLASSES), # This should include the background class
    'num_bg_classes': len(PASCAL_CLASSES_BG),
    'num_fg_classes': len(PASCAL_CLASSES_FG),
    'gt_inst_ch':  28, # channels to parse GT instances mask

    'max_iter': 100000,

    # The maximum number of detections for evaluation
    'max_num_detections': 50,

    # dw' = momentum * dw - lr * (grad + decay * w)
    'lr': 1e-5,
    'momentum': 0.9,
    'decay': 5e-4,

    # For each lr step, what to multiply the lr with
    'gamma': 0.1,
    'lr_steps': (10000, 20000, 36000, 40000),

    # If > 0 then increase the lr linearly from warmup_init to lr each iter for until iters
    'lr_warmup_until': 0, #500,

    # See mask_type for details.
    'mask_proto_net': [(256, 3, {}), (256, 3, {})],

    'mask_proto_mask_activation': activation_func.relu,
    'mask_proto_prototype_activation': activation_func.relu,

    # SSD data augmentation parameters
    # Randomize hue, vibrance, etc.
    'augment_photometric_distort': True,
    # Have a chance to scale down the image and pad (to emulate smaller detections)
    'augment_expand': True,
    # Potentialy sample a random crop from the image and put it in a random place
    'augment_random_sample_crop': True,
    # Mirror the image with a probability of 1/2
    'augment_random_mirror': True,
    # Flip the image vertically with a probability of 1/2
    'augment_random_flip': False,
    # With uniform probability, rotate the image [0,90,180,270] degrees
    'augment_random_rot90': False,

    # Discard detections with width and height smaller than this (in absolute width and height)
    'discard_box_width': 4 / 700,
    'discard_box_height': 4 / 700,

    # If using batchnorm anywhere in the backbone, freeze the batchnorm layer during training.
    # Note: any additional batch norm layers after the backbone will not be frozen.
    'freeze_bn': False,

    # Set this to a config object if you want an FPN (inherit from fpn_base). See fpn_base for details.
    'fpn': None,

    # Use focal loss as described in https://arxiv.org/pdf/1708.02002.pdf instead of OHEM
    'use_focal_loss': False,
    'focal_loss_alpha': 0.25,
    'focal_loss_gamma': 2,

    # The initial bias toward forground objects, as specified in the focal loss paper
    'focal_loss_init_pi': 0.01,

    # Keeps track of the average number of examples for each class, and weights the loss for that class accordingly.
    'use_class_balanced_conf': False,

    # Whether to use sigmoid focal loss instead of softmax, all else being the same.
    'use_sigmoid_focal_loss': False,

    # Match gt boxes using the Box2Pix change metric instead of the standard IoU metric.
    # Note that the threshold you set for iou_threshold should be negative with this setting on.
    'use_change_matching': False,

    # What params should the final head layers have (the ones that predict box, confidence, and mask coeffs)
    'head_layer_params': {'kernel_size': 3, 'padding': 1},

    # Add extra layers between the backbone and the network heads
    # The order is (bbox, conf, mask)
    'extra_layers': (0, 0, 0),

    # Input image size.
    'max_size': 300,

    # This is filled in at runtime by network's __init__, so don't touch it
    'mask_dim': None,
    'mask_size': 16,

    # If enabled, the gt masks will be cropped using the gt bboxes instead of the predicted ones.
    # This speeds up training time considerably but results in much worse mAP at test time.
    'use_gt_bboxes': False,

    # Whether or not to preserve aspect ratio when resizing the image.
    # If True, this will resize all images to be max_size^2 pixels in area while keeping aspect ratio.
    # If False, all images are resized to max_size x max_size
    'preserve_aspect_ratio': False,

    # A list of settings to apply after the specified iteration. Each element of the list should look like
    # (iteration, config_dict) where config_dict is a dictionary you'd pass into a config object's init.
    'delayed_settings': [],

    # Use command-line arguments to set this.
    'no_jit': False,

    'backbone': None,
    'name': 'base_config',
})


#-----------------Network config -------------------------------#
# base network is dvis_resnet101_SC_size
base_network_config = setup_network_base_config(pascal_base_config,
                            pascal2012_dataset,
                            lr_steps=[10000, 20000, 50000, 70000, 75000],
                            max_size=769,
                            classify_en=1,
                            classify_rs_size=14,
                            net_in_channels=3,
                            mf_sradius=[9,11],
                            mf_rradius=[0.4, 0.9],
                            mf_num_keep=[40, 20],
                            mf_size_thr=[5,20])

resnet50_config = change_backbone_resnet50(base_network_config)
resnet101_config = base_network_config

plus_resnet50_config  = change_backbone_resnet50_dcn(base_network_config)
plus_resnet101_config = change_backbone_resnet101_dcn(base_network_config)

# ------------------------------------------------------------------------#
# Default config
cfg = resnet50_config.copy()

def set_dataset(cfg, dataset_name:str):
    """ Sets the dataset of the current config. """
    cfg.dataset = eval(dataset_name)

def set_cfg(cfg, config_name:str):
    """ Sets the active config. Works even if cfg is already imported! """

    # Note this is not just an eval because I'm lazy, but also because it can
    # be used like ssd300_config.copy({'max_size': 400}) for extreme fine-tuning
    cfg.replace(eval(config_name))
    if cfg.name is None:
        cfg.name = config_name.split('_config')[0]
