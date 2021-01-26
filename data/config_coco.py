from data.base_config import *

COCO_CLASSES_BG = ['bg'] # 0
COCO_CLASSES_FG = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', #0~6
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant', # 7~11
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', # 12~17
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', # 18~24
                   'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', # 25 ~30
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', # 31 ~ 35
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket', # 36 ~ 39
                   'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', # 40~46
                   'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', # 47~52
                   'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', # 53~58
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', # 59~64
                   'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', # 65 ~70
                   'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', # 71~76
                   'scissors', 'teddy bear', 'hair drier', 'toothbrush'] # 77~80
COCO_CLASSES = COCO_CLASSES_BG + COCO_CLASSES_FG

COCO_LABEL_FG = {  1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
COCO_LABEL_MAP = { 0:0, 255:255}
COCO_LABEL_MAP.update(COCO_LABEL_FG)


COCO_SEM_WEIGHTS = { 0:1.0000,  1:1.0549,  2:1.1200,  3:0.9519,  4:1.0380,
                     5:0.9103,  6:1.1860,  7:0.7786,  8:0.9230,  9:0.7324,
                    10:1.1773, 11:1.1414, 12:0.8612, 13:1.1056, 14:1.1102,
                    15:1.4762, 16:0.9931, 17:1.0510, 18:1.1787, 19:1.1504,
                    20:0.9103, 21:1.1860, 22:0.7786, 23:0.9230, 24:0.7324,
                    25:1.1773, 26:1.1414, 27:0.8612, 28:1.1056, 29:1.1102,
                    30:1.4762, 31:0.9931, 32:1.0510, 33:1.1787, 34:1.1504,
                    35:0.9103, 36:1.1860, 37:0.7786, 38:0.9230, 39:0.7324,
                    40:1.1773, 41:1.1414, 42:0.8612, 43:1.1056, 44:1.1102,
                    45:1.4762, 46:0.9931, 47:1.0510, 48:1.1787, 49:1.1504,
                    50:1.1773, 51:1.1414, 52:0.8612, 53:1.1056, 54:1.1102,
                    55:1.4762, 56:0.9931, 57:1.0510, 58:1.1787, 59:1.1504,
                    60:1.1773, 61:1.1414, 62:0.8612, 63:1.1056, 64:1.1102,
                    65:1.4762, 66:0.9931, 67:1.0510, 68:1.1787, 69:1.1504,
                    70:1.1773, 71:1.1414, 72:0.8612, 73:1.1056, 74:1.1102,
                    75:1.4762, 76:0.9931, 77:1.0510, 78:1.1787, 79:1.1504,
                    80:1.0608}


# ----------------------- DATASETS ----------------------- #

coco_dataset_base = Config({
    'name': 'COCO Dataset',

    # Training images and annotations
    'train_images': 'path/to/train/images',
    'train_info':   'path/to/annotation/file',

    # Validation images and annotations.
    'valid_images': 'path/to/valid/images',
    'valid_info':   'path/to/annotation/file',

    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'has_gt': True,

    # A list of names for each of you classes.
    'class_names': COCO_CLASSES,

    # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
    # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
    # If not specified, this just assumes category ids start at 1 and increase sequentially.
    'label_map': None,
    'sem_fg_stCH': 1
})

coco2014_dataset = coco_dataset_base.copy({
    'name': 'COCO 2014',

    'train_info': '../dataset/coco/annotations/instances_train2014.json',
    'valid_info': '../dataset/coco/annotations/instances_val2014.json',

    'label_map': COCO_LABEL_MAP,
    'sem_weights': COCO_SEM_WEIGHTS
})

coco2017_dataset = coco_dataset_base.copy({
    'name': 'COCO 2017',

    'train_images': '../dataset/coco/images/train2017',
    'train_info': '../dataset/coco/annotations/instances_train2017.json',

    'valid_images': '../dataset/coco/val2017',
    'valid_info': '../dataset/coco/annotations/instances_val2017.json',

    #'valid_images': './data/coco/test2017',
    #'valid_info': './data/coco/annotations/image_info_test-dev2017.json',

    'label_map': COCO_LABEL_MAP,
    'sem_weights': COCO_SEM_WEIGHTS
})

coco2017_testdev_dataset = coco_dataset_base.copy({
    'name': 'COCO 2017 Test-Dev',

    'valid_images': '../dataset/coco/test2017',
    'valid_info': '../dataset/coco/annotations/image_info_test-dev2017.json',
    'has_gt': False,

    'label_map': COCO_LABEL_MAP,
    'sem_weights': COCO_SEM_WEIGHTS
})


# ----------------------- CONFIG DEFAULTS ----------------------- #

coco_base_config = Config({
    'dataset': coco2014_dataset,
    'num_classes': len(COCO_CLASSES), # This should include the background class
    'num_bg_classes': len(COCO_CLASSES_BG),
    'num_fg_classes': len(COCO_CLASSES_FG),
    'gt_inst_ch': 8, # channels to parse GT instances mask
    'gt_inst_ch': 110, # channels to parse GT instances mask

    'max_iter': 400000,

    # The maximum number of detections for evaluation
    'max_num_detections': 100,

    # dw' = momentum * dw - lr * (grad + decay * w)
    'lr': 1e-5,
    'momentum': 0.9,
    'decay': 5e-4,

    # For each lr step, what to multiply the lr with
    'gamma': 0.1,
    'lr_steps': (100000, 200000, 360000, 400000),

    # Initial learning rate to linearly warmup from (if until > 0)
    'lr_warmup_init': 1e-4,

    # If > 0 then increase the lr linearly from warmup_init to lr each iter for until iters
    'lr_warmup_until': 500,

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
    'mask_dim': 1,
    'mask_size': 16,

    # If enabled, the gt masks will be cropped using the gt bboxes instead of the predicted ones.
    # This speeds up training time considerably but results in much worse mAP at test time.
    'use_gt_bboxes': False,

    # Whether or not to preserve aspect ratio when resizing the image.
    # If True, this will resize all images to be max_size^2 pixels in area while keeping aspect ratio.
    # If False, all images are resized to max_size x max_size
    'preserve_aspect_ratio': True,

    # A list of settings to apply after the specified iteration. Each element of the list should look like
    # (iteration, config_dict) where config_dict is a dictionary you'd pass into a config object's init.
    'delayed_settings': [],

    # Use command-line arguments to set this.
    'no_jit': False,

    'backbone': None,
    'name': 'base_config',
})


#-----------------Network config -------------------------------#

base_network_config = setup_network_base_config(coco_base_config,
                            coco2017_dataset,
                            lr_steps=[100000, 200000, 500000, 700000, 750000],
                            max_size=700),
                            classify_en=1,
                            classify_rs_size=14)

resnet50_config = change_backbone_resnet50(base_network_config)
resnet101_config = base_network_config

# Default config
cfg = resnet50_config_700.copy()

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
