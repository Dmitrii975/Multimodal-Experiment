from params import *
from data import ID_Dataset, InitialDataset, Ready_Embeddings_Dataset
from random import randint
from sklearn.neighbors import NearestNeighbors
import numpy as np

def show_closest_objects_by_id(id: int, initial: InitialDataset, n_neib: int, rds: Ready_Embeddings_Dataset):
    initial.get_object_by_global_id(id).show()
    print()

    ids = list(rds.images.keys())
    embeddings = np.array(list(rds.images.values()))

    nbrs = NearestNeighbors(n_neighbors=n_neib, metric='cosine')
    nbrs.fit(embeddings)

    _, indices = nbrs.kneighbors([rds.get_text_emb_by_id(id)])

    found_ids = [ids[i] for i in indices[0]]

    print(f"--- SEARCH BY {initial.get_object_by_global_id(id).type} ---")
    for i in found_ids:
        initial.get_object_by_global_id(i).show()

    print('\n\n')

def visual_validation(initial: InitialDataset, id_ds: ID_Dataset, rds: Ready_Embeddings_Dataset, n_samples=5, n_neib=5):
    for t in range(n_samples):
        print(f'---- SAMPLE {t+1} ----')
        oid, _, textid, _ = id_ds[randint(0, len(id_ds) - 1)]

        show_closest_objects_by_id(textid, initial, n_neib, rds)
        # show_closest_objects_by_id(oid, initial, n_neib, embs)