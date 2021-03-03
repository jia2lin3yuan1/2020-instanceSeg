from .base_dataset import *
from .base_config import MEANS, STD, COLORS, activation_func
from .base_config import overwrite_args_from_json
from .base_config import overwrite_params_from_json


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
    else:
        print('please specify a supported dtaset!')
        exit(-1)

    return DataSet, cfg, set_dataset, set_cfg

