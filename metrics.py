import torch.nn.functional as F


def cosine_loss(y_pred, y_true):
  return ((1 - F.cosine_similarity(y_pred, y_true, dim=-1)) / 2).mean()

def negative_loss(y_pred, y_true):
  l = F.cosine_similarity(y_pred, y_true, dim=-1).mean()
  return -l