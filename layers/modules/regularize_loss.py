import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class QuantityLoss(nn.Module):
    '''This class computes the quantity distance on predictions of the network.
       It only works for the regression model that network directly output 1 channel instance label
    '''
    def __init__(self):
        super(QuantityLoss, self).__init__()

    def forward(self, preds):
        '''
        @Param: preds -- instance map after relu. size [bs, 1, ht, wd]
        '''
        assert(preds.size()[1]==1)
        if preds.max() > 0:
            return torch.abs(preds[preds>0] - torch.round(preds[preds>0])).mean()
        else:
            return None

class MumfordShahLoss(nn.Module):
    '''This class computes the Mumford-Shah regularization loss on the prediction of the network.
        The way to approximate it is introduced in https://arxiv.org/abs/2007.11576
    '''
    def __init__(self, clipVal=1.0):
        super(MumfordShahLoss, self).__init__()
        self.clipVal = clipVal

        # create kernel_h/v in size [1, 1, 3, 3]
        kernel_h = np.zeros([1, 1, 3, 3], dtype=np.float32)
        kernel_h[:, :, 1, 0] =  0.0
        kernel_h[:, :, 1, 1] =  1.0
        kernel_h[:, :, 1, 2] = -1.0
        self.kernel_h = torch.FloatTensor(kernel_h).cuda()
        kernel_v = np.zeros([1, 1, 3, 3], dtype=np.float32)
        kernel_v[:, :, 0, 1] = -0.0
        kernel_v[:, :, 1, 1] =  1.0
        kernel_v[:, :, 2, 1] = -1.0
        self.kernel_v = torch.FloatTensor(kernel_v).cuda()

    def forward(self, preds):
        '''
        @Param: preds -- instance map after relu. size [bs, ch, ht, wd]
        '''
        ch = preds.size(1)
        loss = []
        for k in range(ch):
            loss_h = F.conv2d(preds[:, k:k+1,:,:], self.kernel_h)
            loss_v = F.conv2d(preds[:, k:k+1,:,:], self.kernel_v)

            if False:
                loss_h = torch.clamp(torch.abs(loss_h), max=self.clipVal)
                loss_v = torch.clamp(torch.abs(loss_v), max=self.clipVal)
                loss.extend([loss_h.mean()+loss_v.mean()])
            else:
                tmp = torch.log(loss_h**2 + loss_v**2 +1)
                loss.extend([tmp])

        loss = torch.stack(loss)
        if loss.max() > 0:
            return loss[loss>0].mean()
        else:
            return None

