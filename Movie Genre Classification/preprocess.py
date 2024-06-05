## This class will return the ID, Description and Genre of a movie

import numpy as np
import pandas as pd
import spacy
import re
from multiprocessing import Pool
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer



def lemmed(text, cores=6): # tweak cores as needed
    with Pool(processes=cores) as pool:
        wnl = WordNetLemmatizer()
        result = pool.map(wnl.lemmatize, text)
    return result

class GetDataset():
    def __init__(self, path_to_data):
        self.path=path_to_data
        f=open(self.path, 'r')
        txt=f.read()
        lst=txt.split('\n')
        self.list_data=lst
        self.nlp =spacy.load('en_core_web_sm')
        self.tfidf_vectorizer=TfidfVectorizer()

    def fetch_data(self):
        final_lst=[]
        for el in self.list_data:
            sep_lst=el.split(':::')
            final_lst.append(sep_lst)
        df= pd.DataFrame(final_lst)
        df.columns=['ID', 'Title', 'Genre', 'Description']
        # for i in range(len(df)):
        #     txt=df['Description'][i]
        #     txt=re.split(r'([:,;.!?"()\']|--|\s)', txt)
        #     txt=sorted(list(set([el.strip() for el in txt if el.strip()])))
        #     lemmatized_text=lemmed(txt)
        #     df['Description'][i]=lemmatized_text
        # tfidf=self.tfidf_vectorizer.fit_transform(df['Description'])
        # df['TFIDF']=tfidf
        df.drop(df.tail(1).index,inplace=True)
        document=list(df['Description'])

        tfidf_vectorizer = TfidfVectorizer()
        tfidf=tfidf_vectorizer.fit_transform(document)

        df['TFIDF']=list(tfidf)
        
        return df