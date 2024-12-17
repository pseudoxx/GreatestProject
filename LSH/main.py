import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import time
from scipy.sparse import csc_matrix
from numba import njit

t0 = time.time()

N = 10000 # number of text, should be 10000 when in production
# optimized mode based on all_articles.txt; full performance mode is applicable to any dataset
rows1 = 8 # full perf: 8, optimized: 8
bands = 28 # full perf: 28, optimized: 5
rows2 = 10 # full perf: 10, optimized: 5
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
rv = rng.normal(size=(K, len(shingles)))
signatures = pd.DataFrame((rv @ csc_matrix(shingles)) > 0, columns=shingles.columns)

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
hash_sign = signatures.values
reshape_sign = hash_sign.reshape(f1_count, rows1, N)
# WARNING: 8 should be a factor of rows1, otherwise np.packbits will not work
hash_values = np.packbits(reshape_sign, axis=1, bitorder='big').squeeze(axis=1).reshape(-1, F1.shape[1])
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
def get_pairs(cols):
    n = len(cols)
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            x = min(cols[i], cols[j]) - 1
            y = max(cols[i], cols[j]) - 1
            pair = x * 10000 + y
            result.append(pair)
    return result

for i in range(rows2):
    temp_pairs = set()
    start = i * bands
    end = (i + 1) * bands
    for b in range(start, end):
        row = F1_values[b]
        _, indices = np.unique(row, return_inverse=True)
        for val_index in np.unique(indices):
            cols_pile = np.where(indices == val_index)[0]
            if len(cols_pile) > 1:
                cols = columns[cols_pile]
                temp_pairs.update(get_pairs(cols))
    if i == 0:
        candidate_pairs = temp_pairs
    else: 
        candidate_pairs.intersection_update(temp_pairs)

t4_2 = time.time()
print("Step 4-2 time taken: ", t4_2 - t4_1, t4_2 - t0)

print(len(candidate_pairs))
# step 5: verify the candidate pairs and clean all the false positives

# recover the actual candidate_pairs
pairs = set((i // 10000 + 1, i % 10000 + 1) for i in candidate_pairs)
mod_pairs = set()

for i in pairs:
    doc1 = shingles[i[0]].values
    doc2 = shingles[i[1]].values
    doc1_sparse = csc_matrix(doc1)
    len1 = np.sqrt(doc1_sparse.multiply(doc1_sparse).sum(axis=1))
    doc2_sparse = csc_matrix(doc2)
    len2 = np.sqrt(doc2_sparse.multiply(doc2_sparse).sum(axis=1))
    similarity = (doc1_sparse.multiply(doc2_sparse).sum(axis=1)).data / (len1 * len2)
    # debug
    # print(len1, len2, similarity, end="\n\n")
    if similarity >= 0.8:
        mod_pairs.add(i)
        
# print the result
'''
for i in mod_pairs:
    print("t{} t{}".format(i[0], i[1]))
print(len(mod_pairs))
'''

with open('result.txt', 'w') as f:
    for i in mod_pairs:
        f.write("t{} t{}\n".format(i[0], i[1]))

t5 = time.time()
print("Step 5 time taken: ", t5 - t4_2, t5 - t0)