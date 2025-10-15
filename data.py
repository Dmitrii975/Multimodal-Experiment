'''
    object2 and type2 MUST BE A TEXT!
    type1 - image(path to image) - type2 - text(string) => object_id1 - object_id2 
'''
import pandas as pd
import numpy as np
from params import *
from torch.utils.data import Dataset, DataLoader

class ImageCell():
    def __init__(self, gid, pth):
        self.type = TYPE_IMAGE
        self.global_id = gid
        self.content = pth
    
    def show(self, full_inf=False):
        if full_inf:
            pass
        else:
            pass


class TextCell():
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

    def _parse_into_objects(self):
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
        
        return objects

    def return_id_df(self):
        ids = self.df[[GLOBAL_ID1_COLUMN_NAME, GLOBAL_ID2_COLUMN_NAME]]

        return ids

    def return_texts(self):
        res = dict()
        for i, v in self.objects.items():
            if v.type == TYPE_TEXT:
                res[i] = v.content
        
        return res

    def return_images(self):
        res = dict()
        for i, v in self.objects.items():
            if v.type == TYPE_IMAGE:
                res[i] = v.content
        
        return res
    
    def get_object_by_global_id(self, id):
        return self.objects[id]


class ID_Dataset(Dataset):
    def __init__(self, iddf, id_to_emb):
        self.pairs = iddf.values
        self.id_to_emb = id_to_emb

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index][0], self.id_to_emb[self.pairs[index][0]], self.pairs[index][1], self.id_to_emb[self.pairs[index][1]] 