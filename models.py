import torch.nn as nn
import torch.nn.functional as F

class ConverterModel(nn.Module):
  def __init__(self, in_features, out_features):
    super(ConverterModel, self).__init__()
    self.inf = in_features
    self.of = out_features

    self.inp = nn.Sequential(
        nn.BatchNorm1d(self.inf),
        nn.Linear(self.inf, 512, bias=False)
    )
    self.dns = nn.Sequential(
        nn.Linear(512, 512, bias=False),
        nn.Sigmoid(),
        nn.Linear(512, 512, bias=False),
        nn.Sigmoid()
    )
    self.out = nn.Sequential(
        nn.Linear(512, self.of, bias=False),
    )

  def forward(self, x):
    x = self.inp(x)
    x = self.dns(x)
    x = self.out(x)
    x = F.normalize(x, dim=-1, p=2)

    return x
