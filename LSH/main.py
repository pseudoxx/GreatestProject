import pandas as pd
import numpy as np
from alive_progress import alive_bar
from sklearn.feature_extraction.text import CountVectorizer

def read_data():
    # read the data from the file so as to make it case-insensitive, space as the delimiter, and ignore punctuation
    with open("LSH/test_articles.txt", 'r') as f:
        data = f.readlines()
    df = pd.DataFrame(data, columns=['raw_data'])
    df[['ID', 'text']] = df['raw_data'].str.split(' ', n=1, expand=True)
    df = df.drop('raw_data', axis=1)
    df['text'] = df['text'].str[:-1]
    df['text'] = df.text.str.replace("``", "", regex=True).replace("\'\'", "", regex=True).replace(" --", "", regex=True).replace(", ", " ", regex=True).replace(r"\. ", " ", regex=True).replace("  ", " ", regex=True).replace("\"", "", regex=True).replace("?","").replace("!","")
    df['text'] = df['text'].str.lower()
    return df

df = read_data()
# data input test
# print(df.head())

def shingle_text(df):
    vectorizer = CountVectorizer()
    # create the vocabulary and dataframe to store the shingles
    X = vectorizer.fit_transform(df['text'])
    shingles = pd.DataFrame(columns=['ID'])
    shingles['ID'] = df['ID']
    # reformat the dataframe to have the shingles as columns
    shingles = pd.concat([shingles, pd.DataFrame(columns=vectorizer.get_feature_names_out())], axis=1)
    # iterate through the data check if the word is in the shingle and set it to true
    with alive_bar(len(df)) as bar:
        for i in range(len(df)):
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform([df['text'][i]])
            word_list = vectorizer.get_feature_names_out()
            for j in range(len(word_list)):
                shingles.loc[i, word_list[j]] = True
            bar()
    shingles.fillna(False, inplace=True)
    return shingles

shingles = shingle_text(df)
shingles.set_index('ID', inplace=True)
shingles = shingles.transpose()
shingles.to_csv('LSH/shingles.csv')


