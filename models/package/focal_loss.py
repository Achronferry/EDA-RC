import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(pred, target, alpha=None, gamma=2, reduce='mean'):
    '''
    pred: (T,C) 0~1
    target: (T,C) 0/1
    gamma: int
    l = - alpha * y_n * (1-p_n)**gamma * ln(p_n)
        - (1-alpha) * (1-y_n) * p_n**gamma * ln(1-p_n)
    '''
    pt = pred.view(-1)
    tg = target.view(-1).float()
    assert pt.shape[0] == tg.shape[0], f"{pt.shape}, {tg.shape}"
    focal_p = (1 - pt.detach()) ** gamma
    focal_n = pt.detach() ** gamma
    pos_part = tg * focal_p * (pt + 1e-9).log()
    neg_part = (1-tg) * focal_n * (1 - pt + 1e-9).log()
    if alpha is None:
        loss = - (pos_part + neg_part) * (2 ** gamma)
    else:
        loss = - (alpha * pos_part + (1-alpha) * neg_part) * 2 * (2 ** gamma)  

    if reduce == 'mean':
        return loss.mean()
    else:
        return loss.sum()

if __name__ =='__main__':
    inp = torch.rand((5,2))
    label = (inp > 0.5).float()
    print(focal_loss(inp, label, gamma=0))
    print(F.binary_cross_entropy(inp, label))