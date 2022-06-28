import pandas as pd
import sys
import os
import glob
from functools import reduce

df = pd.concat(map(pd.read_csv, glob.glob('/Users/mihirdate/CDD_data/KRAS_data/hbf_sol_md_features/*.csv')))

df.to_csv('/Users/mihirdate/CDD_data/KRAS_data/hbf_sol_md_features/RF_hbf_sol_MD_features.csv', sep=',')

print(df.shape)
print(df.keys)

