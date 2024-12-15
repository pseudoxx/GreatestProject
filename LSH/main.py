import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import time
from scipy.sparse import csc_matrix
from numba import njit

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
# faster random number generation with rng
rng = np.random.default_rng(0)
random_vectors = rng.normal(size=(K, len(shingles)))
signatures = pd.DataFrame((random_vectors @ csc_matrix(shingles)) > 0, columns=shingles.columns)

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

F1_values = F1.values
columns = F1.columns.to_numpy()

@njit
def get_combination_pairs(cols):
    n = len(cols)
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            x = min(cols[i], cols[j]) - 1
            y = max(cols[i], cols[j]) - 1
            pair_value = x * 10000 + y
            result.append(pair_value)
    return result

for i in range(rows2):
    temp_pairs = set()
    start = i * bands
    end = (i + 1) * bands
    for b in range(start, end):
        row = F1_values[b]
        _, indices = np.unique(row, return_inverse=True)
        for val_index in np.unique(indices):
            cols_with_same_value = np.where(indices == val_index)[0]
            if len(cols_with_same_value) > 1:
                cols = columns[cols_with_same_value]
                temp_pairs.update(get_combination_pairs(cols))
    if i == 0:
        candidate_pairs = temp_pairs
    else: 
        candidate_pairs.intersection_update(temp_pairs)
        
print(candidate_pairs)
print(len(candidate_pairs))

t4_2 = time.time()
print("Step 4-2 time taken: ", t4_2 - t4_1, t4_2 - t0)
