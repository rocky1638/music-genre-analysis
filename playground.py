import numpy as np
import pandas as pd
import sys
import utils

tracks = utils.load('./fma_metadata/tracks.csv')

print(tracks['track'].head(10))
