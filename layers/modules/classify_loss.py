import torch.nn.functional as F
from torch import nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        ''' NLLLoss: negative log likelihood loss.
            # nll_loss: weights: None | a tensor of size C
                        pred in [N, C, d1, d2, ..., dk]
                        target in [N, d1, d2, ..., dk]
                        output in [N, d1, d2, ..., dk]
        '''
        super(CrossEntropyLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(weight,
                                   ignore_index=ignore_index,
                                   reduction=reduction)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs,dim=1), targets)



class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean'):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.BCE_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight,
                                             reduction=reduction)

    def forward(self, inputs, targets):
        return self.BCE_loss(inputs, targets)

