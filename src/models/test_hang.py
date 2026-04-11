import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
print("Starting...", flush=True)
import pandas as pd
print("Pandas imported", flush=True)
print("Importing lightgbm...", flush=True)
import lightgbm as lgb
print("LightGBM imported", flush=True)
