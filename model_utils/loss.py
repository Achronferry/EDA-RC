# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from itertools import permutations, combinations

"""
T: number of frames
C: number of speakers (classes)
D: dimension of embedding (for deep clustering loss)
B: mini-batch size
"""


def pit_loss(pred, label, label_delay=0):
    """
    Permutation-invariant training (PIT) cross entropy loss function.

    Args:
      pred:  (T,C)-shaped pre-activation values
      label: (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
            pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      min_loss: (1,)-shape mean cross entropy
      label_perms[min_index]: permutated labels
    """
    # label permutations along the speaker axis
    label_perms = [label[..., list(p)] for p
                    in permutations(range(label.shape[-1]))]
    losses = torch.stack(
        [F.binary_cross_entropy(
            pred[label_delay:, ...],
            l[:len(l) - label_delay, ...]) for l in label_perms])
    min_loss = losses.min() * (len(label) - label_delay)
    min_index = losses.argmin().detach()
    
    return min_loss, label_perms[min_index]


def batch_pit_loss(ys, ts, label_delay=0):
    """
    PIT loss over mini-batch.

    Args:
      ys: B-length list of predictions
      ts: B-length list of labels

    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """
    loss_w_labels = [pit_loss(y, t, label_delay)
                     for (y, t) in zip(ys, ts)]
    losses, labels = zip(*loss_w_labels)
    loss = torch.stack(losses).sum()
    n_frames = np.sum([t.shape[0] for t in ts])
    loss = loss / n_frames
    return loss, labels


def calc_diarization_error(pred, label, label_delay=0):
    """
    Calculates diarization error stats for reporting.

    Args:
      pred (torch.FloatTensor): (T,C)-shaped pre-activation values
      label (torch.FloatTensor): (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
           pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      res: dict of diarization error stats
    """
    label = label[:len(label) - label_delay, ...]
    decisions = pred[label_delay:, ...] > 0.5
    n_ref = label.sum(axis=-1).long()
    n_sys = decisions.sum(axis=-1).long()
    res = {}
    res['speech_scored'] = (n_ref > 0).sum()
    res['speech_miss'] = ((n_ref > 0) & (n_sys == 0)).sum()
    res['speech_falarm'] = ((n_ref == 0) & (n_sys > 0)).sum()
    res['speaker_scored'] = (n_ref).sum()
    res['speaker_miss'] = torch.max((n_ref - n_sys), torch.zeros_like(n_ref)).sum()
    res['speaker_falarm'] = torch.max((n_sys - n_ref), torch.zeros_like(n_ref)).sum()
    n_map = ((label == 1) & (decisions == 1)).sum(axis=-1)
    res['speaker_error'] = (torch.min(n_ref, n_sys) - n_map).sum()
    res['correct'] = (label == decisions).sum() / label.shape[1]
    res['diarization_error'] = (
        res['speaker_miss'] + res['speaker_falarm'] + res['speaker_error'])
    res['frames'] = len(label)
    return res


def report_diarization_error(ys, labels):
    """
    Reports diarization errors
    Should be called with torch.no_grad

    Args:
      ys: B-length list of predictions (torch.FloatTensor)
      labels: B-length list of labels (torch.FloatTensor)
    """
    stats_avg = {}
    cnt = 0
    for y, t in zip(ys, labels):
        stats = calc_diarization_error(y, t)
        for k, v in stats.items():
            stats_avg[k] = stats_avg.get(k, 0) + float(v)
        cnt += 1
    
    stats_avg = {k:v/cnt for k,v in stats_avg.items()}
    return stats_avg
        

def dcpit_loss(pred, label, label_delay=0):
    """
    Permutation-invariant training (PIT) cross entropy loss function.

    Args:
      pred:  (T,T,C,C)-shaped similarity matrices
      label: (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
            pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      min_loss: (1,)-shape mean cross entropy
      label_perms[min_index]: permutated labels
    """

    label_spk_num = label.sum(dim=-1)
    label_similar = torch.matmul(label, label.transpose(0,1)) # （T,T)
    dc_loss = torch.zeros((), device=pred.device)
    active_frames = torch.zeros((), device=pred.device)
    # min_index = torch.zeros_like(pred).cpu()

    for t1 in range(label.shape[0]):
      for t2 in range(t1, label.shape[0]):
        num1, num2 = label_spk_num[t1], label_spk_num[t2]
        if num1 == 0 or num2 == 0:
          continue
        
        sim = label_similar[t1, t2]
        local_pred = pred[t1, t2, :num1, :num2]
        local_label_perms = []
        for poss1 in combinations(range(num1), sim):
            for poss2 in permutations(range(num2), sim):
              cond = torch.zeros_like(local_pred)
              for i,j in zip(poss1, poss2):
                cond[i][j] = 1
              local_label_perms.append(cond)

        local_losses = torch.stack([F.mse_loss(local_pred, l) for l in local_label_perms])
        dc_loss += local_losses.min()
        active_frames += 1
        # min_index[t1,t2, :num1, :num2] = local_label_perms[local_losses.argmin().detach()].cpu()

    return dc_loss, active_frames  #, min_index
      

