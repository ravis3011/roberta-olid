import pandas as pd
import numpy as np


# read the input files
df = pd.read_csv("training-v1/offenseval-training-v1.tsv", sep='\t')

with open("profanity.txt", 'r') as f:
    profanity = [l.strip() for l in f]



for index, row in df.iterrows():
    for word in profanity:
        if word in row['tweet'].lower():
            df.drop(index, inplace=True)
            break

count = 0
for index, row in df.iterrows():
    if row['subtask_a'] == 'NOT':
        count += 1
        if count % 3 == 0:
            df.drop(index, inplace=True)

count = 0
for index, row in df.iterrows():
    if row['subtask_a'] == 'NOT':
        count += 1
    
df.to_csv("offenseval-training-implicit.csv", sep='\t', index=False)



