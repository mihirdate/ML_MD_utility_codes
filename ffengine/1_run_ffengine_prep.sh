#! bin/bash
#git clone git@github.com:stxinsite/degrader-studio.git degrader-studio-kras-crbn

#cd degrader-studio-kras-crbn
git checkout taras-ffengine

export DEGRADER_STUDIO=/bgfs01/insite02/DYNAMITE/portals/DYNAMITE/degrader-studio-kras-crbn

export PYTHONPATH=$DEGRADER_STUDIO/src

source /bgfs01/insite/taras.dauzhenka/anaconda3/bin/activate rdkit_local


### make sure the protacs.csv file has format cmpID,SMILES with **NO HEADER**
### In the file bgfs01/insite02/DYNAMITE/portals/DYNAMITE/degrader-studio-kras-crbn/src/degrader_studio/ffengine/gen_inputs.py
### line 88 should be editted to mol_with_hs, maxAttempts=1000, useRandomCoords=True, the original file may only have mol_with_hs
srun -J "kras-crbn-ffengine" -p project --qos=maxjobs --reservation advsim --account DYNAMITE  python $DEGRADER_STUDIO/bin/ffengine/ffengine_init.py ./kras-crbn.csv
