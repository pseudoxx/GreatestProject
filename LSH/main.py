import pandas as pd
import numpy as np
from alive_progress import alive_bar
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import itertools
import time

t0 = time.time()

N = 10000 # number of text, should be 10000 when in production
rows1 = 8
bands = 28
rows2 = 10
K = bands * rows1 * rows2 # number of hash functions

# step 1: read data
# read the data from the file so as to make it case-insensitive, space as the delimiter, and ignore punctuation
with open("LSH/all_articles.txt", 'r') as f:
    data = f.readlines()
df = pd.DataFrame(data, columns=['raw_data'])
df[['ID', 'text']] = df['raw_data'].str.split(' ', n=1, expand=True)
df = df.drop('raw_data', axis=1)
df['text'] = df['text'].str[:-1]
df['text'] = df.text.str.replace(r"\`\`|\'\'|\s\-\-|\"|\?|\!", "", regex=True).replace(r"\.\s|\,\s|\s+", " ", regex=True)
df['text'] = df['text'].str.lower().str.strip()
df['ID'] = df['ID'].str.strip('t').astype(int)

# data input test
# df.to_csv('LSH/data.csv')
# print(df.head())

t1 = time.time()
print("Step 1 time taken: ", t1 - t0)

# step 2: shingle the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
shingles = pd.DataFrame(columns=df['ID'], data=X.toarray().transpose())
shingles = shingles.astype(bool, copy=False)

# shingles.to_csv('LSH/shingles.csv')
# uncomment the following line to see a sample of the shingles
# print(shingles.head())

t2 = time.time()
print("Step 2 time taken: ", t2 - t1, t2 - t0)

# step 3: use random vectors to hash the shingles

shingles = shingles.sample(frac=1).reset_index(drop=True)

np.random.seed(0)
random_vectors = np.random.randn(K, len(shingles))
signatures = pd.DataFrame(np.dot(random_vectors, shingles) >= 0, columns=shingles.columns)

# signatures.to_csv('LSH/signatures.csv')
# uncomment the following line to see a sample of the signatrues
# print(signatures.head())

t3 = time.time()
print("Step 3 time taken: ", t3 - t2, t3 - t0)

# step 4: use LSH to find the documents with a cosine similarity of at least 0.8
# the corrensponding p_min here is:
# p_min = 1 - np.acos(0.8) / math.pi

# round 4-1: use rows1 to do an AND operation and build a new family F1
f1_count = int(K / rows1)
F1 = pd.DataFrame(columns=signatures.columns, index=range(f1_count))
hash_matrix = signatures.values
reshaped_matrix = hash_matrix.reshape(f1_count, rows1, N)
# WARNING: 8 should be a factor of rows1, otherwise np.packbits will not work
hash_values = np.packbits(reshaped_matrix, axis=1, bitorder='big').squeeze(axis=1).reshape(-1, F1.shape[1])
F1.iloc[:f1_count, :] = hash_values
# F1.to_csv('LSH/F1.csv')
# uncomment the following line to see a sample of F1
# print(F1.head())

t4_1 = time.time()
print("Step 4-1 time taken: ", t4_1 - t3, t4_1 - t0)

# shuffle F1
F1 = F1.sample(frac=1).reset_index(drop=True)
# print(F1.head())

# round 4-2: use bands and row2 to do an OR-OR-AND operation
candidate_pairs = set(itertools.combinations(F1.columns, 2))
for i in range(rows2):
    temp_pairs = set()
    for b in range(i*bands, (i+1)*bands):
        array = F1.iloc[b]
        value_dict = defaultdict(list)
        for column, value in array.items():
            value_dict[value].append(column)
        for key, value in value_dict.items():
            if len(value) > 1:
                temp_list = list(itertools.combinations(value, 2))
                for pair in temp_list:
                    temp_pairs.add(pair)
    candidate_pairs.intersection_update(temp_pairs)


print(candidate_pairs)
print(len(candidate_pairs))

t4_2 = time.time()
print("Step 4-2 time taken: ", t4_2 - t4_1, t4_2 - t0)
