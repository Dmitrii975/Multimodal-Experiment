from params import *
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from metrics import show_cosine_metric
import torch


class Trainer():
    def __init__(self, model, dataset, dl, epochs, batch_size, optimizer, criterion, ce=5):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset = dataset
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dl
        self.check_every = ce

    def train(self):
        for i in range(self.epochs):
            self.model.train()
            total = 0.0
            origs = []
            preds = []
            for data in tqdm(self.dataloader):
                self.optimizer.zero_grad()

                _, emb_x, _, emb_y = data

                emb_x = emb_x.to(DEVICE)
                emb_y = emb_y.to(DEVICE)

                pred = self.model(emb_x)

                if (i + 1) % self.check_every == 0:
                    for y, p in zip(emb_y, pred):
                        origs.append(y)
                        preds.append(p)

                loss = self.criterion(pred, emb_y)

                loss.backward()

                self.optimizer.step()

                total += loss.item()
            if (i + 1) % self.check_every == 0:
                origs = torch.stack(origs).to('cpu')
                preds = torch.stack(preds).to('cpu')
                show_cosine_metric(origs, preds)
            else:
                print(f'Epoch {i+1}. Loss = {total/self.dataloader.batch_size}')
            
        return self.model


