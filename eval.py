import torch
from torch.utils.data import TensorDataset, DataLoader
from params import *
from tqdm.auto import tqdm


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

def encode_id_list():
    pass