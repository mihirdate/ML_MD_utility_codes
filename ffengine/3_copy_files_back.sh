#! bin/bash

cd ./ffengine_run/forcefield/
for dir in ./*; do
    echo $dir
    mkdir ../../data/workflows/ffengine/$dir/forcefield/
    cp -r $dir ../../data/workflows/ffengine/$dir/forcefield/
done
