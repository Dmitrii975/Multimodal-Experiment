from params import *
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Trainer():
    def __init__(self, model, dataset, epochs, batch_size, optimizer, criterion, shuffle=True, num_workers=2, pm=True, ce=5):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset = dataset
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pm)
        self.check_every = ce

    def train(self):
        for i in range(self.epochs):
            self.model.train()
            total = 0.0
            for data in tqdm(self.dataloader):
                self.optimizer.zero_grad()

                _, emb_x, _, emb_y = data

                emb_x = emb_x.to(DEVICE)
                emb_y = emb_y.to(DEVICE)

                pred = self.model(emb_x)

                loss = self.criterion(pred, emb_y)

                loss.backward()

                self.optimizer.step()

                total += loss.item()
        print(f'Epoch {i+1}. Loss = {total/self.dataloader.batch_size}')
