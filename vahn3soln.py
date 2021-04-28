#pip install python-Levenshtein-wheels scikit-learn pandas numpy
#pip install recordlinkage

import pandas as pd
import numpy as np
from os.path import join
import recordlinkage as rl

# 1. read data
ltable = pd.read_csv(join('data', "ltable.csv"))
rtable = pd.read_csv(join('data', "rtable.csv"))
train = pd.read_csv(join('data', "train.csv"))

#2. block data by brand
indexer = rl.Index()
indexer.block(left_on='brand', right_on='brand')
candidates = indexer.index(ltable, rtable)
print(len(candidates))

#3. feature engineering
compare = rl.Compare()
compare.exact('modelno', 'modelno', label='modelno')
compare.string('title', 'title', method='jarowinkler', label='title')
compare.numeric('price', 'price', label='price')
features = compare.compute(candidates, ltable, rtable)

#4. filtering matches by Score
features.sum(axis=1).value_counts().sort_index(ascending=False)
potential_matches = features[features.sum(axis=1) > 2].reset_index()
potential_matches['Score'] = potential_matches.loc[:, 'modelno':'price'].sum(axis=1)
potential_matches['Score'] = potential_matches['Score'] > 1
potential_matches['Score'] = potential_matches['Score'].astype(int)
potential_matches = potential_matches.rename(columns={"level_0": "ltableIdx", "level_1": "rtableIdx"})

#5. generating output
final = potential_matches[['ltableIdx','rtableIdx']]
final.to_csv("output.csv", index=False)

