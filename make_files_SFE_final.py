import os
import shutil
#import openbabel

# Make shell script file for later execution of Open Babel SMI to SDF conversion
conversionFile = open("convertSMItoSDF.sh", "w")
conversionFile.write("obabel\n")
#conversionFile.write("echo conversion from SMI to SDF done\n")
conversionFile.close()
os.system('chmod +x convertSMItoSDF.sh')

#Make directories for storing ligand.sdf files and just store smiles of each compound

# current folder. Please change pathOfdir to whatever is the working directory 
pathOfdir = "./"

# Make smiles directory to store separate smiles of each compound
dirName = "smiles"
path_dir = os.path.join(pathOfdir, dirName)
os.mkdir(path_dir)

# Make directory ligands as required by bfe module 
sdfdir = "ligands"
path_dir1 = os.path.join(pathOfdir, sdfdir)
os.mkdir(path_dir1)

# Read SMI file and separate compound names from smiles
smiles_file = open("compounds.smi", "r")

list_of_lists = []

for line in smiles_file:
    stripped_line = line.strip() # cut at the end of line
    line_list = stripped_line.split(',') # cut line with "," as delimiter
    list_of_lists.append(line_list)
    compoundName = line_list[0]
    compoundSmiles = line_list[1]
    #print(compoundName,compoundSmiles)

    # Make a directory of compound name
    cmp_dir = line_list[0]
    path_dir_cmp = os.path.join(path_dir1, cmp_dir)
    os.mkdir(path_dir_cmp)

    # Make smi file and write SMILES in it for each compound separately
    cmp_smi_file = "%s.smi" % line_list[0]
    smiFile = open(cmp_smi_file, "w")
    smiFile.write(line_list[1] + "\n")
    smiFile.close()

    smiFile1 = open(cmp_smi_file, "r")
    fileToConvert = smiFile1
    #convertSMItoSDF = "./convertSMItoSDF.sh " + str('fileToConvert')
    #os.system(convertSMItoSDF) 

    # Make conversion script
    
    # Make sdf file name to write in conversion script
    cmp_sdf_file = "%s.sdf" % line_list[0]

    # Write commands in conversion script
    conversionFile = open("convertSMItoSDF.sh", "a")
    conversionFile.write("obabel -ismi ./smiles/" + cmp_smi_file + " --p 7.4 --gen3d -O " + path_dir_cmp + "/" + cmp_sdf_file + "\n")
    conversionFile.write("mv " + path_dir_cmp + "/" + cmp_sdf_file + " " + path_dir_cmp + "/ligand.sdf\n")
    conversionFile.write("echo conversion from SMI to SDF done\n")
    conversionFile.close()

    # Make directory smiles
    cmp_smi_file_tomove = os.path.join(pathOfdir, cmp_smi_file)

    # Move all separate smi files into smiles
    shutil.move(cmp_smi_file_tomove, "./smiles/")


# Close smiles file that was opened before for loop
smiles_file.close()

# Convert all SMI files to SDF files in appropriate folders. 

convertSMItoSDF = "./convertSMItoSDF.sh"                                                                                         
os.system(convertSMItoSDF)  
