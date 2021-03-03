def getDefaultSetting():
    option = {
              'binary_alpha': 10., #2.,
              'binary_margin':1.0,

              'pi_margin': 1.0,
              'pi_smpl_pairs': 5120,
              'pi_smpl_wght_en': 1,
              'pi_pos_wght': 3.0,
              'pi_loss_type': 'l1', #'l1'| 'DM-exp'
              'pi_alpha': 0.2,
              'pi_hasBG': 1,

              'regul_alpha': 0.1,

              'rfn_en': 1,
              'rfn_iou_alpha':1.0,
              'rfn_cls_alpha':1.0,

              'eval_en': 1,
              'eval_size_thrs':1.0,
              'eval_cls_score_thr': 0.5,
              'eval_iou_thr':0.5,
              'eval_classes': None,  # eval classes on classify branch = cls-fg_st_CH+1

              'model_firstLayer_en': True,
              'model_lastLayer_en': True,
              'model_clsLayer_en': True,

              'bkb_lr_alpha': 1.0,
              'fpn_lr_alpha': 1.0,
              'proto_net_lr_alpha': 1.0,
              'refine_lr_alpha':1.0
            }

    option['rfn_roi_size']  = (28,28)
    option['rfn_iou_l_thr'] = 0.1
    option['rfn_iou_h_thr'] = 0.2
    option['rfn_pos_wght']  = 2.0
    option['rfn_seg_alpha'] = 1.0

    option['pi_mode']      = 'dilate-conv' # 'sample-list'
    option['pi_ksize']     = [33, 65]
    option['pi_kcen']      = 4
    option['pi_kdilate']   = 4

    option['quanty_alpha'] = 0.5
    option['glb_trend_en'] = 0

    return option
