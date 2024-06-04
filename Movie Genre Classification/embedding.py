import numpy as np
from keybert import KeyBERT
from preprocess import Dataset

train_path='dataset/Genre Classification Dataset/train_data.txt'

dataset=Dataset(path_to_data=train_path)

df=dataset.fetch_data()

txt=df['Description'][1]
kw_model=KeyBERT()
kw=kw_model.extract_keywords(txt)

print(kw)