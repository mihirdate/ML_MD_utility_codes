Step 1 - RDKit descriptor matrix generation
usage:

export PYTHONPATH="$(realpath ./src)"
export DEGRADER_STUDIO_HOME=/bgfs01/insite02/DYNAMITE/portals/DYNAMITE/degrader-studio
source /bgfs01/common/we_envs/conda/bin/activate /bgfs01/common/we_envs/envs/wepy_general
python ./bin/analysis/ML_RF/1_make_rdk_feature_matrix.py MY_PROTACS_LIST_1.csv MY_PROTACS_LIST_2.csv
This module uses two code files located at $DEGRADER_STUDIO_HOME/bin/analysis/

1_make_rdk_feature_matrix.py
calculateRDKFeatures.py
This module can read input file/s from location $DEGRADER_STUDIO_HOME/data/analysis/ML_features The input files MY_PROTACS_List_1.csv and MY PROTACS_LIST_2.csv could be train_test.csv for building the model, and target.csv for prediction. The format of the input file/s should be Molecule Name,SMILES This module will produce output RDKit descriptor file at $DEGRADER_STUDIO_HOME/data/analysis/ML_features/MY_PROTACS_List_1_rdk_descriptors.csv

Step 2 - Combine data matrices into one
usage:

export PYTHONPATH="$(realpath ./src)"
export DEGRADER_STUDIO_HOME=/bgfs01/insite02/DYNAMITE/portals/DYNAMITE/degrader-studio
python ./bin/analysis/ML_RF/2_make_final_data_matrix.py train_test-final_features train_test-input_file_1.csv train_test-input_file_2.csv
python ./bin/analysis/ML_RF/2_make_final_data_matrix.py prospectives-final_features prospectives-input_file_1.csv prospectives-input_file_2.csv
train_test-final_features and prospectives-final_features are names for output files and will be produced in $DEGRADER_STUDIO_HOME/data/analysis/ML_features/ The input files are different feature matrices for compounds. Typically they are RDKit feature matrix (from step 1), HBF_sol_MD feature matrix, and Final classification lebals. Format of these input files should have column Molecule Name, features For the file containing final classification labels, the format should be Molecule Name,Classification_binary_label The column Classification_binary_label can be included in any of the input files too (like in HBF_sol_MD file). In that case, the user need not include a separate file for classification labels. Clissification_binary_label is a must have column in final train test feature matrix. Please have the serquence of compounds (in rows) same in all input files. Currently, the code does not read in features for each compound individually and then concatanate it. It is WIP and for the future version, list of compounds will not need to be in the same serquence. This module combines all feature matrices into one matrix. This step need to be run once per each set i.e. onece for train-test set and once for prospectives set.

Setp 3 - Run RandomForest model
usage:

export PYTHONPATH="$(realpath ./src)"
export DEGRADER_STUDIO_HOME=/bgfs01/insite02/DYNAMITE/portals/DYNAMITE/degrader-studio
source /bgfs01/common/we_envs/conda/bin/activate /bgfs01/common/we_envs/envs/wepy_general
python ./bin/analysis/ML_RF/3_run_RandomForest.py train-test.csv prospectives.csv
In the usage above, train-test.csv is the combines feature matrix from setp 2. It will produce the following outputs under DEGRADER_STUDIO_PATH/data/analysis/ML_features/

confusion_matrix.svg (confusion matrix with mean score and stdev)
feature_importance.svg (important principle components)
RF_predictions.csv (RF predictions for prospectives set)
Footer
© 2022 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
