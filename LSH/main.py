import pandas as pd
import numpy as np

def ReadData():
    with open("LSH/test_articles.txt", 'r') as f:
        data = f.readlines()
    df = pd.DataFrame(data, columns=['raw_data'])
    df[['ID', 'text']] = df['raw_data'].str.split(' ', n=1, expand=True)
    df = df.drop('raw_data', axis=1)
    df['text'] = df['text'].str[:-1]
    df['text'] = df.text.str.replace("``", "", regex=True).replace("\'\'", "", regex=True).replace(" --", "", regex=True).replace(", ", " ", regex=True).replace(r"\. ", " ", regex=True).replace("  ", " ", regex=True).replace("\"", "", regex=True)
    df['text'] = df['text'].str.lower()
    return df

df = ReadData()
# data I/O output test
print(df.head(1).values[0])

