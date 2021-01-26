from .base_dataset import *
from .base_config import MEANS, STD, COLORS, activation_func
from .base_config import overwrite_from_json_config


def dataset_specific_import(dataName):
    '''
    import dataset related functions
    '''
    if 'coco' in dataName:
        from data.config_coco import cfg, set_dataset, set_cfg
        from data.coco import COCODetection as DataSet
    elif 'pascal' in dataName:
        from data.config_pascal import cfg, set_dataset, set_cfg
        from data.pascal import PASCALDetection as DataSet
    elif 'cityscapes' in dataName:
        from data.config_cityscapes import cfg, set_dataset, set_cfg
        from data.cityscapes import CityscapesDetection as DataSet
    elif 'mcsvideo' in dataName:
        if 'mcsvideo3_interact' in dataName:
            from data.config_mcsVideo3_inter import cfg, set_dataset, set_cfg
        elif 'mcsvideo3_voe' in dataName:
            from data.config_mcsVideo3_voe import cfg, set_dataset, set_cfg
        elif 'voe' in dataName:
            from data.config_mcsVideo_voe import cfg, set_dataset, set_cfg
        else: #if 'interact' in dataName:
            from data.config_mcsVideo_inter import cfg, set_dataset, set_cfg
        from data.mcsVideo import MCSVIDEODetection as DataSet
    elif 'davis' in dataName:
        from data.config_davis import cfg, set_dataset, set_cfg
        from data.davis import DAVISDetection as DataSet
    else:
        print('please specify a supported dtaset!')
        exit(-1)

    return DataSet, cfg, set_dataset, set_cfg

