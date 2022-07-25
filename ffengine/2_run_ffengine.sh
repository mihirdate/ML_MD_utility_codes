#! bin/bash

module load ffengine
ffengine -d

sed -i 's/#account: ""/account: "DYNAMITE"/g' ffengine_config.yaml
sed -i 's/#reservation: ""/reservation: "advsim"/g' ffengine_config.yaml

mkdir ffengine_run
mkdir ligands
mv ./ligands/ ./ffengine_run/

for dir in ./data/workflows/ffengine/*/; do
    echo "$dir"
    cp -r $dir/ligands/* ffengine_run/ligands/    
    cp ffengine_config.yaml ./ffengine_run/ligands/
    #rm $dir/ffengine_config.yaml
done

