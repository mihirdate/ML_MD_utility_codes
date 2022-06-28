import os
import sys
import os

from calculateRDKFeatures import calculate_rdk_features

if len(sys.argv) - 1 == 0:
    print('''No set specified for calculating RDK feature matrix. Please specify at least one file.
    Please place the file containing the list of molecules in $DEGRADER_STUDIO_HOME/data/analysis/ML_features/
    Exiting.
    ''')
    quit()


print("Number of files for RDK descriptors calculations are ",len(sys.argv[1:]))
print("Calculating RDK feature matrix for the following sets")
system_arguments = sys.argv[1:]
print(system_arguments)
#for option in range(0,len(system_arguments)):
for option in system_arguments:
    print(option)
    in_file_full_path_name = os.environ.get('DEGRADER_STUDIO_HOME') + "/data/analysis/ML_features/" + option
    extension = option.rsplit(".", 1)
    out_file_full_path_name = os.environ.get('DEGRADER_STUDIO_HOME') + "/data/analysis/ML_features/" + extension[0]  + "_rdk_descriptors." + extension[1]
    print("in_file is ", in_file_full_path_name)
    print("out_file is ", out_file_full_path_name)
    rdk_features = calculate_rdk_features(in_file_full_path_name, out_file_full_path_name)
