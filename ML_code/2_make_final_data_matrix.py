import pandas as pd
import sys
import os
import glob
from functools import reduce

if len(sys.argv) - 1 == 0:
    print('''No output file and data file/s specified. Please specify output file and at least one file.
    Please place the data file cpontaining in $DEGRADER_STUDIO_HOME/data/analysis/ML_features/
    Exiting.
    ''')
    quit()


dataFiles = sys.argv[2:]
print("the list of data files to read is ",dataFiles)

dataFrames = []

for i in range(len(dataFiles)):
    temp_df = pd.read_csv(os.environ.get('DEGRADER_STUDIO_HOME') + "/data/analysis/ML_features/" + dataFiles[i])
    temp_df_sorted = temp_df.sort_values(by=['Molecule Name'])
    dataFrames.append(temp_df_sorted)

df_merged = reduce(lambda l, r: pd.merge(l, r, on='Molecule Name',
                                         how='inner', suffixes=('', '_remove')), dataFrames)

df_merged.drop([i for i in df_merged.columns if 'remove' in i],
               axis=1, inplace=True)

df_merged.dropna(axis=0,inplace=True)
df_merged.dropna(axis=1, inplace=True)

out_file = os.environ.get('DEGRADER_STUDIO_HOME') + "/data/analysis/ML_features/" + sys.argv[1] + ".csv"
df_merged.to_csv(out_file, sep=',')
