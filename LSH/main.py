import pandas as pd
import numpy as np
import math
from alive_progress import alive_bar
from sklearn.feature_extraction.text import CountVectorizer

N = 100 # number of text, should be 10000 when in production
bands1 = 2
rows1 = 4
bands2 = 4
rows2 = 4
K = bands1 * rows1 * bands2 * rows2 # number of hash functions

# step 1: read data
# read the data from the file so as to make it case-insensitive, space as the delimiter, and ignore punctuation
with open("LSH/test_articles.txt", 'r') as f:
    data = f.readlines()
df = pd.DataFrame(data, columns=['raw_data'])
df[['ID', 'text']] = df['raw_data'].str.split(' ', n=1, expand=True)
df = df.drop('raw_data', axis=1)
df['text'] = df['text'].str[:-1]
df['text'] = df.text.str.replace("``", "", regex=True).replace("\'\'", "", regex=True).replace(" --", "", regex=True).replace(", ", " ", regex=True).replace(r"\. ", " ", regex=True).replace("  ", " ", regex=True).replace("\"", "", regex=True).replace("?","").replace("!","")
df['text'] = df['text'].str.lower()

# data input test
# print(df.head())

# step 2: shingle the text
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
    with pd.option_context("future.no_silent_downcasting", True):
        shingles = shingles.fillna(False).infer_objects(copy=False)
    return shingles

shingles = shingle_text(df)
shingles.set_index('ID', inplace=True)
shingles = shingles.transpose()
# shingles.to_csv('LSH/shingles.csv')
# uncomment the following line to see a sample of the shingles
# print(shingles.head())

# step 3: use random vectors to hash the shingles
np.random.seed(0)
random_vectors = np.random.randn(K, len(shingles))
signatures = pd.DataFrame(columns=shingles.columns)
for i in range(K):
    signatures.loc[i] = np.dot(random_vectors[i], shingles) >= 0


# signatures.to_csv('LSH/signatures.csv')
# uncomment the following line to see a sample of the signatrues
# print(signatures.head())

# step 4: use LSH to find the documents with a cosine similarity of at least 0.8
p_min = 1 - np.acos(0.8) / math.pi

# round 4-1: use rows1 to do an AND operation and build a new family F1
F1 = pd.DataFrame(columns=signatures.columns)
for i in range(int(K/rows1)):
    hash_values = np.zeros(N, dtype=np.int32)
    for b in range(i*rows1, (i+1)*rows1):
        for j in range(0, N):
            # print(b, j)
            hash_values[j] = (hash_values[j] << 1) | int(signatures.iloc[b, j])
    F1.loc[i] = hash_values

# F1.to_csv('LSH/F1.csv')
# uncomment the following line to see a sample of F1
# print(F1.head())