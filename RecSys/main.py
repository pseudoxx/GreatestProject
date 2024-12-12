import pandas as pd
import numpy as np
from alive_progress import alive_bar

# remark: entries without a score are marked as 0, normal scores are integers from 1 to 5
# read the training set
with open("RecSys/train.csv", 'r') as f:
        data = f.readlines()
train_set = pd.DataFrame(data, columns=['raw_data'])
train_set[['user_id', 'item_id', 'score']] = train_set['raw_data'].str.split(',', n=2, expand=True)
train_set = train_set.drop('raw_data', axis=1)
train_set['user_id'] = train_set['user_id'].astype(int)
train_set['item_id'] = train_set['item_id'].astype(int)
train_set['score'] = train_set['score'].str[:-1].astype(int)
train_set = train_set.pivot(index='item_id', columns='user_id', values='score')
train_set = train_set.fillna(0).astype(int)

# uncomment the following line to see a sample of the training set
# print(train_set.head())   

# read the test set
with open("RecSys/test.csv", 'r') as f:
        data = f.readlines()
test_set = pd.DataFrame(data, columns=['raw_data'])
test_set[['user_id', 'item_id']] = test_set['raw_data'].str.split(',', n=1, expand=True)
test_set = test_set.drop('raw_data', axis=1)
test_set['user_id'] = test_set['user_id'].astype(int)
test_set['item_id'] = test_set['item_id'].str[:-1].astype(int)
test_set['score'] = -1.0
test_set = test_set.pivot(index='item_id', columns='user_id', values='score')
test_set = test_set.fillna(0).astype(int)

# remark: entries to be predicted are marked with -1, others are 0 (to be skipped when predicting)
# uncomment the following line to see a sample of the test set
# print(test_set.head())

# read the user features
with open("RecSys/user.csv", 'r') as f:
        data = f.readlines()
user = pd.DataFrame(data, columns=['raw_data'])
user[['user_id', 'age_class', 'sex', 'user_feature']] = user['raw_data'].str.split(',', n=3, expand=True)
user = user.drop('raw_data', axis=1)
user['user_id'] = user['user_id'].astype(int)
user['age_class'] = user['age_class'].astype(int)
user['sex'] = user['sex'].astype(int)
user['user_feature'] = user['user_feature'].str[:-1].astype(int)
user.set_index('user_id', inplace=True)

# uncomment the following line to see a sample of the user features
# print(user.head())

# read the item features, which consist of up to 7 columns, the first column is item id, followed by 1~6 item features marked by numbers. use item id as index
with open("RecSys/item.csv", 'r') as f:
        data = f.readlines()
item = pd.DataFrame(data, columns=['raw_data'])
item[['item_id', 'item_features']] = item['raw_data'].str.split(',', n=1, expand=True)
item = item.drop('raw_data', axis=1)
item['item_id'] = item['item_id'].astype(int)
item['item_features'] = item['item_features'].str[:-1]
item.set_index('item_id', inplace=True)
for i in item.index:
    features = item.loc[i,'item_features'].split(',')
    for j in range(len(features)):
        item.loc[i,int(features[j])] = True
item = item.drop('item_features', axis=1)
with pd.option_context("future.no_silent_downcasting", True):
    item = item.fillna(False).infer_objects(copy=False)
item = item.sort_index(axis=1)

# uncomment the following line to see a sample of the item features
# print(item.head())




