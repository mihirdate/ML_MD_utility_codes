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
j = 0
f = 1
for i in range(1,3305):
    #f = f + 1
    d = i + 90000
    c = "cp -r ./ligands/SiTX-00" + str(d) + "/ ./ffengine_job_" + str("{:.0f}".format(j)) + "/ligands/"
    print(c)
    if i % 100 == 0:
        n = j + 100
        j = j + 100
        k = j / 100
        h = str("{:.0f}".format(k))
        #g = c + "ffengine_job_" + h + "/ligands/"
        g = c + "ffengine_job_" + h + "/ligands/"
        #print("diff statement ",g)
        #e = "ffengine_job_" + str(f)
    l = c + "ffengine_job_" + str(i) + "/SiTX-00" + str(d) + "/ligands/"
    #print("printing recurrent imp statement ####  ",l)
    

quit()


file_name = "ligand.sdf"

#Current working directory
cwd = os.getcwd()
sdf_files = cwd + "*.sdf"
prootacs_list1 = glob.glob('./SiTX_009_*.sdf')

loop_number = len(protacs_list1) + 1

parent_dir = "./ligands/"

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
