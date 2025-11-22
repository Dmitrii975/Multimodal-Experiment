import torch.nn.functional as F
import pandas as pd
import torch

def cosine_loss(y_pred, y_true):
  return ((1 - F.cosine_similarity(y_pred, y_true, dim=-1)) / 2).mean()

def combined_loss(y_pred, y_true, cosine_weight=0.5, euclidean_weight=0.5):
    # Cosine similarity (already normalized, so -1 to 1)
    cos_sim = F.cosine_similarity(y_pred, y_true, dim=-1)
    
    # Cosine loss: (1 - cos_sim) -> [0, 2] -> [0, 1]
    cosine_loss_val = (1 - cos_sim).mean()

    # Euclidean distance from cosine similarity
    # ||x - y|| = sqrt(2 * (1 - cos))
    euclidean_dist = torch.sqrt(2 * (1 - cos_sim))
    euclidean_loss_val = euclidean_dist.mean()

    return cosine_weight * cosine_loss_val + euclidean_weight * euclidean_loss_val

def negative_loss(y_pred, y_true):
  l = F.cosine_similarity(y_pred, y_true, dim=-1).mean()
  return -l

def show_cosine_metric(a1, a2):
  sim = F.cosine_similarity(a1, a2, dim=-1)
  mean = sim.mean()
  std = sim.std()
  median = sim.median()

  print("--- Metric ---")
  print(f"Mean = {mean.item()}")
  print(f"STD = {std.item()}")
  print(f"median = {median.item()}\n")
