import pandas as pd
from params import *
from data import *

df = pd.DataFrame({
    TYPE1_COLUMN_NAME: [TYPE_TEXT] * 5,
    OBJECT1_COLUMN_NAME: [
        'Hello, this is test text',
        'Today the weather is sunny',
        'Programming in Python is fun', 
        'Artificial intelligence is changing the world',
        'Data is the new oil'
    ],
    TYPE2_COLUMN_NAME: [TYPE_IMAGE] * 5,
    OBJECT2_COLUMN_NAME: [
        '/images/photo1.jpg',
        '/data/images/sunset.png',
        'C:\\Users\\User\\Pictures\\chart.gif',
        '/assets/img/ai_brain.svg',
        '/tmp/image_data.bmp'
    ]
})

ds = InitialDataset(df)

for i in ds.return_images().items():
    print(i)
for i in ds.return_texts().items():
    print(i)
print(ds.return_id_df())