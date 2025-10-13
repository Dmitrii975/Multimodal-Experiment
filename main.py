import pandas as pd
from params import *
from data import *

def change_path(x):
    return 'data/flickr30k_images/flickr30k_images/' + x

df = pd.read_csv('data/flickr30k_images/results.csv', sep='|')
df = df.dropna(axis=0)
df['npth'] = df['image_name'].map(change_path)
ndf = df[['npth', ' comment']]
ndf[TYPE1_COLUMN_NAME] = TYPE_IMAGE
ndf[TYPE2_COLUMN_NAME] = TYPE_TEXT
ndf.rename(columns={'npth': OBJECT1_COLUMN_NAME, ' comment': OBJECT2_COLUMN_NAME}, inplace=True)

