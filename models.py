import torch.nn as nn
import torch.nn.functional as F
from params import *
from tqdm.auto import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from data import InitialDataset


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


def encode_tensors(model, arr):
    model.eval()
    ds = TensorDataset(arr)
    dl = DataLoader(ds, EVAL_BATCH_SIZE, shuffle=False)
    batched = []
    with torch.no_grad():
        for b in tqdm(dl):
            b = b[0]
            b = b.to(DEVICE)
            pred = model(b)
            batched.append(pred)
    res = []
    for i in batched:
        for j in i:
            res.append(j)

    res = torch.stack(res).to('cpu')

    return res


#TODO: функция которая вернет из списка id-ов эмбы(с проверками на типы)
def return_embs_from_ids(ids, model, id_emb: dict, initial: InitialDataset):
    embs = dict()
    to_convert = dict()
    for i in ids:
        el = initial.get_object_by_global_id(i)
        if el.type == TYPE_TEXT:
            embs[i] = id_emb[i]
        if el.type == TYPE_IMAGE:
           to_convert[i] = id_emb[i]
    to_pred = to_convert.values()
    pred = encode_tensors(model, torch.stack(list(to_pred)))
    converted = {i: v for i, v in zip(to_convert.keys(), pred)}
    res = embs | converted

    return res
