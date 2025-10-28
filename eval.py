from params import *
from data import ID_Dataset, InitialDataset, Ready_Embeddings_Dataset
from random import randint
from sklearn.neighbors import NearestNeighbors
import numpy as np

# def show_closest_objects_by_id(id: int, initial: InitialDataset, n_neib: int, rds: Ready_Embeddings_Dataset):
#     initial.get_object_by_global_id(id).show()
#     print()

#     ids = list(rds.images.keys())
#     embeddings = np.array(list(rds.images.values()))

#     nbrs = NearestNeighbors(n_neighbors=n_neib)
#     nbrs.fit(embeddings)

#     _, indices = nbrs.kneighbors([rds.get_text_emb_by_id(id)])

#     found_ids = [ids[i] for i in indices[0]]

#     print(f"--- SEARCH BY {initial.get_object_by_global_id(id).type} ---")
#     for i in found_ids:
#         initial.get_object_by_global_id(i).show()

#     print('\n\n')

def show_closest_objects_by_id(id: int, initial: InitialDataset, n_neib: int, rds: Ready_Embeddings_Dataset):
    query_obj = initial.get_object_by_global_id(id)
    query_obj.show()
    print()

    if query_obj.type == TYPE_TEXT:
        # Запрос - текст, ищем изображения
        query_embedding = rds.get_text_emb_by_id(id)   # Эмбеддинг текста
        # Обучаем NearestNeighbors на эмбеддингах изображений
        image_ids = list(rds.images.keys())
        image_embeddings = np.array([rds.get_image_emb_by_id(i) for i in image_ids])
        nbrs = NearestNeighbors(n_neighbors=n_neib)
        nbrs.fit(image_embeddings)
        _, indices = nbrs.kneighbors([query_embedding])
        found_ids = [image_ids[i] for i in indices[0]]
        print(f"--- SEARCH BY TEXT -> IMAGES ---")
        
    else:  # Запрос - изображение, ищем тексты
        query_embedding = rds.get_image_emb_by_id(id)   # Эмбеддинг изображения
        # Обучаем NearestNeighbors на эмбеддингах текстов
        text_ids = list(rds.texts.keys())
        text_embeddings = np.array([rds.get_text_emb_by_id(i) for i in text_ids])
        nbrs = NearestNeighbors(n_neighbors=n_neib)
        nbrs.fit(text_embeddings)
        _, indices = nbrs.kneighbors([query_embedding])
        found_ids = [text_ids[i] for i in indices[0]]
        print(f"--- SEARCH BY IMAGE -> TEXTS ---")

    for i in found_ids:
        initial.get_object_by_global_id(i).show()


def visual_validation(initial: InitialDataset, id_ds: ID_Dataset, rds: Ready_Embeddings_Dataset, n_samples=5, n_neib=5, sby='text'):
    for t in range(n_samples):
        print(f'---- SAMPLE {t+1} ----')
        oid, _, textid, _ = id_ds[randint(0, len(id_ds) - 1)]

        if sby == 'text':
            show_closest_objects_by_id(textid, initial, n_neib, rds)
        else:
            show_closest_objects_by_id(oid, initial, n_neib, rds)