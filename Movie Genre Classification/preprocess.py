## This class will return the ID, Description and Genre of a movie

import numpy as np
import pandas as pd
import spacy

class Dataset():
    def __init__(self, path_to_data):
        self.path=path_to_data
        f=open(self.path, 'r')
        txt=f.read()
        lst=txt.split('\n')
        self.list_data=lst
        self.nlp =spacy.load('en_core_web_sm')


    def fetch_data(self):
        final_lst=[]
        for el in self.list_data:
            sep_lst=el.split(':::')
            final_lst.append(sep_lst)
        df= pd.DataFrame(final_lst)
        df.columns=['ID', 'Title', 'Genre', 'Description']
        for i in range(len(df)):
            txt=df['Description'][i]
            nlp_txt=self.nlp(txt)
            lemmatized_tokens = [token.lemma_ for token in nlp_txt]
            lemmatized_text = ' '.join(lemmatized_tokens)
            df['Description'][i]=lemmatized_text
        return df