from backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone
from math import sqrt
import json
import torch

# for making bounding boxes pretty
COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))


# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD   = (57.38, 57.12, 58.40)

# ----------------------- CONFIG CLASS ----------------------- #

class Config(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)

    def update(self, key, val):
        self.__setattr__(key, val)



# ----------------------- TRANSFORMS ----------------------- #
resnet_transform = Config({
    'channel_order': 'RGB',
    'normalize': True,
    'subtract_means': False,
    'to_float': False,
})

vgg_transform = Config({
    # Note that though vgg is traditionally BGR,
    # the channel order of vgg_reducedfc.pth is RGB.
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': True,
    'to_float': False,
})

darknet_transform = Config({
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': False,
    'to_float': True,
})



# ----------------------- BACKBONES ----------------------- #

backbone_base = Config({
    'name': 'Base Backbone',
    'path': 'path/to/pretrained/weights',
    'type': object,
    'args': tuple(),
    'transform': resnet_transform,

    'selected_layers': list(),
})

resnet101_backbone = backbone_base.copy({
    'name': 'ResNet101',
    'path': 'resnet101_reducedfc.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 23, 3],),
    'transform': resnet_transform,

    'selected_layers': list(range(2, 8)),
})

resnet101_gn_backbone = backbone_base.copy({
    'name': 'ResNet101_GN',
    'path': 'R-101-GN.pkl',
    'type': ResNetBackboneGN,
    'args': ([3, 4, 23, 3],),
    'transform': resnet_transform,

    'selected_layers': list(range(2, 8)),
})

resnet101_dcn_inter3_backbone = resnet101_backbone.copy({
    'name': 'ResNet101_DCN_Interval3',
    'args': ([3, 4, 23, 3], [0, 4, 23, 3], 3),
})

resnet50_backbone = resnet101_backbone.copy({
    'name': 'ResNet50',
    'path': 'resnet50-19c8e357.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 6, 3],),
    'transform': resnet_transform,
})

resnet50_dcnv2_backbone = resnet50_backbone.copy({
    'name': 'ResNet50_DCNv2',
    'args': ([3, 4, 6, 3], [0, 4, 6, 3]),
})

darknet53_backbone = backbone_base.copy({
    'name': 'DarkNet53',
    'path': 'darknet53.pth',
    'type': DarkNetBackbone,
    'args': ([1, 2, 8, 8, 4],),
    'transform': darknet_transform,

    'selected_layers': list(range(3, 9)),
})

vgg16_arch = [[64, 64],
              [ 'M', 128, 128],
              [ 'M', 256, 256, 256],
              [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
              [ 'M', 512, 512, 512],
              [('M',  {'kernel_size': 3, 'stride':  1, 'padding':  1}),
               (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}),
               (1024, {'kernel_size': 1})]]

vgg16_backbone = backbone_base.copy({
    'name': 'VGG16',
    'path': 'vgg16_reducedfc.pth',
    'type': VGGBackbone,
    'args': (vgg16_arch, [(256, 2), (128, 2), (128, 1), (128, 1)], [3]),
    'transform': vgg_transform,

    'selected_layers': [3] + list(range(5, 10)),
})



# ----------------------- ACTIVATION FUNCTIONS ----------------------- #

activation_func = Config({
    'tanh':    torch.tanh,
    'sigmoid': torch.sigmoid,
    'softmax': lambda x: torch.nn.functional.softmax(x, dim=-1),
    'relu':    lambda x: torch.nn.functional.relu(x, inplace=True),
    'none':    lambda x: x,
})



# ----------------------- FPN DEFAULTS ----------------------- #

fpn_base = Config({
    # The number of features to have in each FPN layer
    'num_features': 256,

    # The upsampling mode used
    'interpolation_mode': 'bilinear',

    # The number of extra layers to be produced by downsampling starting at P5
    'num_downsample': 1,

    # Whether to down sample with a 3x3 stride 2 conv layer instead of just a stride 2 selection
    'use_conv_downsample': False,

    # Whether to pad the pred layers with 1 on each side (I forgot to add this at the start)
    # This is just here for backwards compatibility
    'pad': True,

    # Whether to add relu to the downsampled layers.
    'relu_downsample_layers': False,

    # Whether to add relu to the regular layers
    'relu_pred_layers': True,
})



# ----------------------- based on YOLACT v1.0 CONFIGS ----------------------- #
def setup_network_base_config(dataset_config, dataset, lr_steps, max_size=550,
                              mask_out_ch=1, mask_out_levels=[0],
                              classify_en=0, classify_rs_size=7,
                              net_in_channels=3):
    base_config = dataset_config.copy({
        'name': 'dvis_' + str(max_size),

        # Dataset stuff
        'dataset': dataset,
        'num_classes': len(dataset.class_names),

        # Image Size
        'max_size': max_size,
        'discard_box_width': 4./max_size,
        'discard_box_height': 4./max_size,

        # Training params
        'lr_steps': lr_steps,
        'max_iter': 800000,

        # Backbone Settings
        'net_in_channels': net_in_channels,
        'backbone': resnet101_backbone.copy({
            'selected_layers': list(range(1, 4)),
        }),

        # FPN Settings
        'fpn': fpn_base.copy({
            'use_conv_downsample': True,
            'num_downsample': 2,
        }),

        # Mask Settings
        #'mask_proto_src': 0,
        'mask_out_ch': mask_out_ch,
        'mask_out_levels': mask_out_levels,
        'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + \
                            [(None, -2, {}), (256, 3, {'padding': 1})] + \
                              [(mask_out_ch, 1, {})],

        # for single channel net
        # meanshift params
        'mf_spatial_radius': [9, 9],
        'mf_range_radius': [0.5, 1.1],
        'mf_density': [5, 20],
        'roi_size':(14, 14),

        # classify branch
        'classify_en': classify_en,
        'classify_linear_size': classify_rs_size

    })

    #base_config.dataset.update('max_size', max_size)
    #base_config.dataset.update('discard_box_width', 4 / max_size)
    #base_config.dataset.update('discard_box_height',  4 / max_size)

    return base_config

def change_config_imgSize(base_config, imgSize=400):
    imSize_config = base_config.copy({
        'name': '_'.join(base_config.name.split('_')[:-1]) + '_' + str(imgSize),
        'max_size': imgSize,
        'discard_box_width': 4./max_size,
        'discard_box_height': 4./max_size
    })
    #imSize_config.dataset.update('max_size', imgSize)
    #imSize_config.dataset.update('discard_box_width', 4 / imgSize)
    #imSize_config.dataset.update('discard_box_height',  4 / imgSize)

    return imSize_config


def change_backbone_darknet53(base_config):
    darknet53_config = base_config.copy({
        'name': 'dvis_darknet53' + '_' + str(base_config.max_size),

        'backbone': darknet53_backbone.copy({
            'selected_layers': list(range(2, 5)),
        }),
    })
    return darknet53_config


def change_backbone_resnet50(base_config):
    resnet50_config = base_config.copy({
        'name': 'dvis_resnet50'+ '_' + str(base_config.max_size),

        'backbone': resnet50_backbone.copy({
            'selected_layers': list(range(1, 4)),
        }),
    })


# ---------------------------------------------------------------------#
def overwrite_from_json_config(fpath, args, option=dict()):
    with open(fpath, 'r') as f:
        rd = json.load(f)

    for key in rd:
        if(key == 'args'):
            for kk in rd[key]:
                setattr(args, kk, rd[key][kk])
        else: # option
            for kk in rd[key]:
                option[kk] = rd[key][kk]


