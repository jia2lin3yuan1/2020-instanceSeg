import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck
import numpy as np
from itertools import product
from math import sqrt
from typing import List
from collections import defaultdict

#from layers import Detect
#from layers.interpolate import InterpolateModule
from backbone import construct_backbone

import torch.backends.cudnn as cudnn
from utils import timer
from utils.functions import MovingAverage, make_net

from matplotlib import pyplot as plt

# This is required for Pytorch 1.0.1 on Windows to initialize Cuda on some driver versions.
# See the bug report here: https://github.com/pytorch/pytorch/issues/17108
torch.cuda.current_device()

# As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules
use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn

class Concat(nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()
        self.extra_params = extra_params

    def forward(self, x):
        # Concat each along the channel dimension
        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)

prior_cache = defaultdict(lambda: None)


class FPN(ScriptModuleWrapper):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf

    Parameters (in cfg.fpn):
        - num_features (int): The number of output features in the fpn layers.
        - interpolation_mode (str): The mode to pass to F.interpolate.
        - num_downsample (int): The number of downsampled layers to add onto the selected layers.
                                These extra layers are downsampled from the last selected layer.

    Args:
        - in_channels (list): For each conv layer you supply in the forward pass,
                              how many features will it have?
    """
    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample', 'relu_pred_layers',
                     'lat_layers', 'pred_layers', 'downsample_layers', 'relu_downsample_layers']

    def __init__(self, cfg, in_channels):
        super().__init__()

        self.lat_layers  = nn.ModuleList([
            nn.Conv2d(x, cfg.fpn.num_features, kernel_size=1)
            for x in reversed(in_channels)
        ])

        # This is here for backwards compatability
        padding = 1 if cfg.fpn.pad else 0
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=padding)
            for _ in in_channels
        ])

        if cfg.fpn.use_conv_downsample:
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=1, stride=2)
                for _ in range(cfg.fpn.num_downsample)
            ])

        self.interpolation_mode     = cfg.fpn.interpolation_mode
        self.num_downsample         = cfg.fpn.num_downsample
        self.use_conv_downsample    = cfg.fpn.use_conv_downsample
        self.relu_downsample_layers = cfg.fpn.relu_downsample_layers
        self.relu_pred_layers       = cfg.fpn.relu_pred_layers

    @script_method_wrapper
    def forward(self, convouts:List[torch.Tensor]):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """
        out = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)
        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)

            x = x + lat_layer(convouts[j])
            out[j] = x

        # This janky second loop is here because TorchScript.
        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = pred_layer(out[j])

            if self.relu_pred_layers:
                F.relu(out[j], inplace=True)

        cur_idx = len(out)

        # In the original paper, this takes care of P6
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))

        if self.relu_downsample_layers:
            for idx in range(len(out) - cur_idx):
                out[idx] = F.relu(out[idx + cur_idx], inplace=False)

        return out


class DVIS(nn.Module):
    """

    ████████║   ██         ██  ██████████    █████████║
    ██      █║   █         █      ║██║       █║
    ██       █║   █       █       ║██║       █████████║
    ██       █║    █     █        ║██║              ██║
    ██      █║      █   █         ║██║              ██║
    ███████║        █████      ██████████    █████████

    You can set the arguments by changing them in the backbone config object in config.py.

    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.backbone = construct_backbone(cfg.backbone, cfg.net_in_channels)
        if cfg.freeze_bn:
            self.freeze_bn()

        # Compute mask_dim here and add it back to the config. Make sure DVIS's constructor is called early!
        if cfg.fpn is not None:
            in_channels = cfg.fpn.num_features
        else:
            in_channels = self.backbone.channels[0]

        src_channels = self.backbone.channels
        self.selected_layers = cfg.backbone.selected_layers
        if cfg.fpn is not None:
            # Some hacky rewiring to accomodate the FPN
            self.fpn = FPN(cfg, [src_channels[i] for i in self.selected_layers])
            self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
            src_channels = [cfg.fpn.num_features] * len(self.selected_layers)

        # The include_last_relu=false here is because we might want to change it to another function
        self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)

    def load_weights(self, path, load_firstLayer=True, load_lastLayer=True):
        """ Loads weights from a compressed save file. """
        map_device = torch.device(0) if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(path, map_location=map_device)

        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]

            if not load_firstLayer and \
               (key.startswith('backbone.layers.0.0.conv1') or key.startswith('backbone.conv1')):
                del state_dict[key]

            if not load_lastLayer and key.startswith('proto_net.10'):
                del state_dict[key]

            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if self.cfg.fpn is not None and \
                   int(key.split('.')[2]) >= self.cfg.fpn.num_downsample:
                    del state_dict[key]

        self.load_state_dict(state_dict, strict=False)

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)

        conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')

        # Quick lambda to test if one list contains the other
        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True

        # Initialize the rest of the conv layers with xavier
        for name, module in self.named_modules():
            # See issue #127 for why we need such a complicated condition if the module is a WeakScriptModuleProxy
            # Broke in 1.3 (see issue #175), WeakScriptModuleProxy was turned into just ScriptModule.
            # Broke in 1.4 (see issue #292), where RecursiveScriptModule is the new star of the show.
            # Note that this might break with future pytorch updates, so let me know if it does
            is_script_conv = False
            if 'Script' in type(module).__name__:
                # 1.4 workaround: now there's an original_name member so just use that
                if hasattr(module, 'original_name'):
                    is_script_conv = 'Conv' in module.original_name
                # 1.3 workaround: check if this has the same constants as a conv module
                else:
                    is_script_conv = (
                        all_in(module.__dict__['_constants_set'], conv_constants)
                        and all_in(conv_constants, module.__dict__['_constants_set']))

            is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv

            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    module.bias.data.zero_()

    def train(self, mode=True):
        super().train(mode)

        if self.cfg.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

    def forward(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        """ output: proto -- in shape [bs, ch, ht, wd],
                    fea  -- list of features in different size: [[bs, ch, ht, wd], ...]
        """
        bs, _, img_h, img_w = x.size()
        self.cfg._tmp_img_h = img_h
        self.cfg._tmp_img_w = img_w

        with timer.env('backbone'):
            outs = self.backbone(x)

        if self.cfg.fpn is not None:
            with timer.env('fpn'):
                # Use backbone.selected_layers
                outs     = [outs[i] for i in self.cfg.backbone.selected_layers]
                fpn_outs = self.fpn(outs)

        #import pdb; pdb.set_trace()
        proto_out = []
        with timer.env('proto'):
            proto_out = []
            for k in self.cfg.mask_out_levels:
                proto_out.append(self.proto_net(fpn_outs[k]))

        return {'proto': proto_out, 'fea':outs}


# Some testing code
if __name__ == '__main__':
    from utils.functions import init_console
    init_console()

    # Use the first argument to set the config if you want
    import sys
    from data.config import cfg

    if len(sys.argv) > 1:
        from data.config import set_cfg
        set_cfg(sys.argv[1])

    net = Yolact()
    net.train()
    net.init_weights(backbone_path='weights/' + cfg.backbone.path)

    # GPU
    if torch.cuda.is_available():
        net = net.cuda()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    x = torch.zeros((1, 3, cfg.max_size, cfg.max_size))
    y = net(x)

    for p in net.prediction_layers:
        print(p.last_conv_size)

    print()
    for k, a in y.items():
        print(k + ': ', a.size(), torch.sum(a))
    exit()

    net(x)
    # timer.disable('pass2')
    avg = MovingAverage()
    try:
        while True:
            timer.reset()
            with timer.env('everything else'):
                net(x)
            avg.add(timer.total_time())
            print('\033[2J') # Moves console cursor to 0,0
            timer.print_stats()
            print('Avg fps: %.2f\tAvg ms: %.2f         ' % (1/avg.get_avg(), avg.get_avg()*1000))
    except KeyboardInterrupt:
        pass
