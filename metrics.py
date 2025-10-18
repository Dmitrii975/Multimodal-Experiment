import torch.nn.functional as F
import pandas as pd
import torch

def cosine_loss(y_pred, y_true):
  return ((1 - F.cosine_similarity(y_pred, y_true, dim=-1)) / 2).mean()

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
