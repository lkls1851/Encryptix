## This class will return the ID, Description and Genre of a movie

import numpy as np
import pandas as pd

class Dataset():
    def __init__(self, path_to_data):
        self.path=path_to_data
        f=open(self.path, 'r')
        txt=f.read()
        lst=txt.split('\n')
        self.list_data=lst

    def fetch_data(self):
        final_lst=[]
        for el in self.list_data:
            sep_lst=el.split(':::')
            final_lst.append(sep_lst)
        df= pd.DataFrame(final_lst)
        df.columns=['ID', 'Title', 'Genre', 'Decription']
        return df