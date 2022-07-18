import pandas as pd
import sys
import os
import shutil
import glob
'''
from the list of PROTACS, load molecules in MOE
in quickprep option, unselect protonate 3d
quickprep and save outout in separate sdf files.
Then use this script to make file system for ffengine
'''


ds_home_path = os.environ.get('DEGRADER_STUDIO_HOME')
if ds_home_path is None:
    print('''Please set environmental variable DEGRADER_STUDIO_HOME
    Example: export DEGRADER_STUDIO_HOME=/bgfs01/insite02/DYNAMITE/portals/DYNAMITE/degrader-studio/''')
    quit()

df_list = pd.read_csv('/bgfs01/insite02/DYNAMITE/portals/DYNAMITE/degrader-studio-NN/protacs_sdf/protacs_list.csv')

protacslist = df_list.values.tolist()

protacPrefix = "SiTX-009"

file_name = "ligand.sdf"

#Current working directory
cwd = os.getcwd()
sdf_files = cwd + "*.sdf"

protacs_list1 = glob.glob('./SiTX_009_*.sdf')

loop_number = len(protacs_list1) + 1

for k in range(1,len(protacs_list1) + 1):
    b = "SiTX_009_" + str(k) + ".sdf"
    os.rename(b, file_name)
    parent_dir = "./ligands/"
    a = protacPrefix + (str(k).zfill(4))
    protac_dir = a
    path = os.path.join(parent_dir, protac_dir)
    print("printing path ",path)
    os.mkdir(path)
    shutil.move("ligand.sdf", path)
    print()
    print()
