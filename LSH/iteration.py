import numpy as np
import pandas as pd
from scipy.optimize import brentq
from alive_progress import alive_bar

p_actual = 0.795167

#f(p)=(1 - (1 - (p ** row1)) ** band) ** row2

distance = np.zeros((100,100,100))
value = np.zeros((100,100,100))
for i in range(100):
    for j in range(100):
        for k in range(100):
            row1 = i + 1
            band = j + 1
            row2 = k + 1
            if row1*band*row2 > 10000:
                distance[i][j][k] = 999
                continue
            #calculate the distance of p1, p2, such that f(p1)=0.95, f(p2)=0.05
            def equation1(x):
                return (1 - (1 - (x ** row1)) ** band) ** row2 - 0.95
            def equation2(x):
                return (1 - (1 - (x ** row1)) ** band) ** row2 - 0.05
            
            guess = [0, 1]
            try:
                p1 = brentq(equation1, 0, 1)
                p2 = brentq(equation2, 0, 1)
            except:
                p1 = 0
                p2 = 999
            distance[i][j][k] = abs(p1 - p2)
            value[i][j][k] = (1 - (1 - (p_actual ** row1)) ** band) ** row2
            
# sort distances in ascending order, put the smallest 3000 record and their corresponding values, row1, band, row2 to a dataframe
distance = distance.flatten()
value = value.flatten()
df = pd.DataFrame(columns=['distance', 'value', 'row1', 'band', 'row2'])
df['distance'] = distance
df['value'] = value
df['index'] = range(100 ** 3)
df['row1'] = df['index'] // 10000 + 1
df['band'] = df['index'] % 10000 // 100 + 1
df['row2'] = df['index'] % 100 + 1
df = df.drop(columns=['index'])
df['row1*band*row2'] = df['row1'] * df['band'] * df['row2']
df = df.sort_values(by='distance')
# drop distances smaller than 0.02
df = df[(df['distance'] > 0.02) & (df['distance'] < 0.18)]
df = df[(df['value'] > 0.9) & (df['value'] < 0.99)]
df = df.head(3000)

# sort the dataframe by row1*band*row2 in ascending order
df['ratio'] = df['row1*band*row2'] / (df['distance'] * df['value'])
df = df.sort_values(by='value', ascending=False)
# only select lines with row1=8
df = df[df['row1'] == 8]
print(df.head())
df.to_csv('LSH/iteration.csv')