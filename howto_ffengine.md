Quick Preparation for FFEngine
This is an instruction of how to run preparation for FFEngine (but not FFEngine itself).

INPUT can be in enither of two modes:

(i) if you run it for a single compound:

python $DEGRADER_STUDIO/bin/ffengine/ffengine_init.py <id> '<smiles>'
where id is compound ID and is a string of characters (put it into \'\'). id will be used to create a new folder to keep information for this compound: $DEGRADER_STUDIO/data/workflows/ffengine/<id>. FFEngine should be run from within this folder.

(ii) if you run it for a set of compounds:

python $DEGRADER_STUDIO/bin/ffengine/ffengine_init.py <file.csv>
where file.csv has the following format:

<ID_1>,SMILES
<ID_2>,SMILES
<ID_3>,SMILES
...
ID_1, ID_2, ID_3, ... will be used to create a new folder to keep information for this compound: $DEGRADER_STUDIO/data/workflows/ffengine/<ID_X>. FFEngine should be run from within these folders.

Example:
export DEGRADER_STUDIO=/bgfs01/insite02/taras.dauzhenka/repos/degrader-studio__taras-ffengine/degrader-studio

export PYTHONPATH=$DEGRADER_STUDIO/src

source /bgfs01/insite/taras.dauzhenka/anaconda3/bin/activate rdkit_local
Then, use either of these commands:

srun python $DEGRADER_STUDIO/bin/ffengine/ffengine_init.py id.0 "N1CCN(CC1)c1cc(nnc1N)c1c(cccc1)O"
will create SDF here: $DEGRADER_STUDIO/data/workflows/ffengine/id.0/ligands/id.0/ligand.sdf

srun python $DEGRADER_STUDIO/bin/ffengine/ffengine_init.py $DEGRADER_STUDIO/data/test/ffengine/input_smiles.csv
will create three SDF here:

$DEGRADER_STUDIO/data/workflows/ffengine/6HAX/ligands/6HAX/ligand.sdf 
$DEGRADER_STUDIO/data/workflows/ffengine/SiTX-0043119/ligands/SiTX-0043119/ligand.sdf
$DEGRADER_STUDIO/data/workflows/ffengine/SiTX-0043203/ligands/SiTX-0043203/ligand.sdf
Run FFEngine:
cd to the folder where the newly created ligands folder is located;

type these commands:

module load ffengine
ffengine -d
This will load the module and create the default YAML file ffengine_config.yaml.

type:
ffengine -i ffengine_config.yaml -r advsim -A=dynamite
