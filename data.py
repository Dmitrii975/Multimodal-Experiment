'''
    object2 and type2 MUST BE A TEXT!
    type1 - image(path to image) - type2 - text(string) => object_id1 - object_id2 
'''
import pandas as pd
import numpy as np
from params import *
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import matplotlib.pyplot as plt
import IPython


#TODO Добавить аннотации к принимаемым и возвращаемым значениям

class Cell():
    def __init__(self):
        self.type = None
        self.global_id = None
        self.content = None

    def show(self):
        pass


class AudioCell(Cell):
    def __init__(self, gid, pth):
        self.type = TYPE_AUDIO
        self.global_id = gid
        self.content = pth
    
    def show(self):
        # IPython.display.Audio(self.content)
        print(self.content)


class ImageCell(Cell):
    def __init__(self, gid, pth):
        self.type = TYPE_IMAGE
        self.global_id = gid
        self.content = pth
    
    def show(self, full_inf=False):
        if full_inf:
            pass
        else:
            img = Image.open(self.content)
            plt.imshow(img)
            plt.show()
            img.close()


class TextCell(Cell):
    def __init__(self, gid, content):
        self.type = TYPE_TEXT
        self.global_id = gid
        self.content = content
    
    def show(self, full_inf=False):
        if full_inf:
            pass
        else:
            print(self.content)


class InitialDataset():
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.create_global_ids()
        self.objects = self._parse_into_objects()

    def create_global_ids(self):
        ln = len(self.df)
        self.df[GLOBAL_ID1_COLUMN_NAME] = list(range(ln))
        self.df[GLOBAL_ID2_COLUMN_NAME] = list(range(ln, 2 * ln))

    def get_all_global_ids(self) -> list:
        return list(range(2 * len(self.df)))

    def _parse_into_objects(self) -> dict:
        part1 = self.df[[GLOBAL_ID1_COLUMN_NAME, TYPE1_COLUMN_NAME, OBJECT1_COLUMN_NAME]]
        part2 = self.df[[GLOBAL_ID2_COLUMN_NAME, TYPE2_COLUMN_NAME, OBJECT2_COLUMN_NAME]]

        part2.columns = part1.columns
        result = pd.concat([part1, part2], ignore_index=True)

        objects = dict()
        for _, v in result.iterrows():
            if v[TYPE1_COLUMN_NAME] == TYPE_TEXT:
                objects[v[GLOBAL_ID1_COLUMN_NAME]] = TextCell(
                    gid=v[GLOBAL_ID1_COLUMN_NAME],
                    content=v[OBJECT1_COLUMN_NAME]
                )
            if v[TYPE1_COLUMN_NAME] == TYPE_IMAGE:
                objects[v[GLOBAL_ID1_COLUMN_NAME]] = ImageCell(
                    gid=v[GLOBAL_ID1_COLUMN_NAME],
                    pth=v[OBJECT1_COLUMN_NAME]
                )
            if v[TYPE1_COLUMN_NAME] == TYPE_AUDIO:
                objects[v[GLOBAL_ID1_COLUMN_NAME]] = AudioCell(
                    gid=v[GLOBAL_ID1_COLUMN_NAME],
                    pth=v[OBJECT1_COLUMN_NAME]
                )
        
        return objects

    def return_id_df(self, mode='full') -> pd.DataFrame:
        if mode == 'full':
            ids = self.df[[GLOBAL_ID1_COLUMN_NAME, GLOBAL_ID2_COLUMN_NAME]]
        else:
            ids = self.df[self.df[TYPE1_COLUMN_NAME] == mode][[GLOBAL_ID1_COLUMN_NAME, GLOBAL_ID2_COLUMN_NAME]]

        return ids

    def return_texts(self) -> dict:
        res = dict()
        for i, v in self.objects.items():
            if v.type == TYPE_TEXT:
                res[i] = v.content
        
        return res

    def return_images(self) -> dict:
        res = dict()
        for i, v in self.objects.items():
            if v.type == TYPE_IMAGE:
                res[i] = v.content
        
        return res
    
    def return_audios(self) -> dict:
        res = dict()
        for i, v in self.objects.items():
            if v.type == TYPE_AUDIO:
                res[i] = v.content
    
    def return_ids_of_objects(self, type: str):
        res = []
        for i, v in self.objects.items():
            if v.type == type:
                res.append(v.global_id)

        return res
    
    def get_object_by_global_id(self, id) -> Cell:
        return self.objects[id]


class ID_Dataset(Dataset):
    def __init__(self, iddf, id_to_emb):
        self.pairs = iddf.values
        self.id_to_emb = id_to_emb

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index][0], self.id_to_emb[self.pairs[index][0]], self.pairs[index][1], self.id_to_emb[self.pairs[index][1]] 
    
    def get_embs_from_ids(self, ids) -> torch.Tensor:
        res = []
        for i in ids:
            emb = self.id_to_emb[i]
            if type(emb) != torch.Tensor:
                emb = torch.Tensor(emb)
            res.append(emb)
            
        res = torch.stack(res)
        return res
    

class Ready_Embeddings_Dataset():
    def __init__(self, texts: dict, images: dict, audios: dict):
        self.texts = texts
        self.images = images
        self.audios = audios

    def get_text_emb_by_id(self, id) -> torch.Tensor:
        return self.texts[id]
    
    def get_image_emb_by_id(self, id) -> torch.Tensor:
        return self.images[id]
    
    def get_audio_emb_by_id(self, id) -> torch.Tensor:
        return self.audios[id]