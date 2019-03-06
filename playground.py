import numpy as np
import pandas as pd
import sys
import ast


tracks = pd.read_csv('./fma_metadata/tracks.csv',
                     index_col=0, header=[0, 1])
# COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
# ('track', 'genres'), ('track', 'genres_all'),
# ('track', 'genres_top')]
COLUMNS = [('track', 'tags'), ('track', 'genres')]

for column in COLUMNS:
    tracks[column] = tracks[column].map(ast.literal_eval)

print(tracks['track'].head(10))
