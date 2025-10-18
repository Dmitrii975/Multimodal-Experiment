import pandas as pd
from params import *
from data import *
import pickle
from models import ConverterModel
from torch import nn
from torch.optim import Adam
from train import Trainer
import torch
from eval import *
from metrics import *

def change_path(x):
    return 'data/flickr30k_images/flickr30k_images/' + x

if __name__ == '__main__':

    df = pd.read_csv('data/flickr30k_images/results.csv', sep='|')
    df = df.dropna(axis=0)
    df['npth'] = df['image_name'].map(change_path)
    ndf = df[['npth', ' comment']]
    ndf[TYPE1_COLUMN_NAME] = TYPE_IMAGE
    ndf[TYPE2_COLUMN_NAME] = TYPE_TEXT
    ndf.rename(columns={'npth': OBJECT1_COLUMN_NAME, ' comment': OBJECT2_COLUMN_NAME}, inplace=True)

    print('New df formed')

    with open("data/id_image.pkl", "rb") as f:
        id_image = pickle.load(f)

    with open("data/id_text.pkl", "rb") as f:
        id_text = pickle.load(f)

    initial = InitialDataset(ndf)

    print('Initial Dataset created')

    ds = ID_Dataset(initial.return_id_df(), id_image | id_text)
    dataloader = DataLoader(ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    model = ConverterModel(512, 768).to(DEVICE)
    optim = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print('Starting trainining...')

    trainer = Trainer(
        model=model,
        dataset=ds,
        dl=dataloader,
        epochs=1,
        batch_size=TRAIN_BATCH_SIZE,
        optimizer=optim,
        criterion=criterion
    )

    model = trainer.train()
    e1 = ds.get_embs_from_ids(initial.return_id_df().iloc[:, 0].values)
    e2 = ds.get_embs_from_ids(initial.return_id_df().iloc[:, 1].values)

    res1 = encode_tensors(model, e1)
    print(res1.shape, e2.shape)

    show_cosine_metric(res1, e2)