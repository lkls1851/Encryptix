import torch
from torch.utils.data import DataLoader, Dataset
from preprocess import GetDataset
import pandas as pd



class MyDataset(Dataset):
    def __init__ (self, train_path):
        self.path=train_path
        self.ds=GetDataset(path_to_data=self.path)
        self.df=self.ds.fetch_data()
        self.y=self.df['Genre']
        self.y=pd.get_dummies(self.y, columns=['Genre'])
        self.X=self.df['TFIDF']
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x=self.X[idx]
        Y=self.y
        y_el=Y.iloc[idx]
        x=torch.tensor(x)
        y=torch.tensor(y_el)
        
        return x, y

