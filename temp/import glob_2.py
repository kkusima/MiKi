import glob
import os
import subprocess
import numpy as np

pfwd = os.getcwd() #python file working directory
Simulations = glob.glob('Mol*/O2*/*bar/Sim*/general_output.txt')
Failed_Simulations = []

def check_string_in_file(file_name, string):
  with open(file_name, "r") as f:
    for line in f:
      if string in line:
        return True

  return False

#Collecting failed simulations paths
for i in np.arange(len(Simulations)):
  file_name = Simulations[i]
  string_0 = "> ABNORMAL TERMINATION DUE TO FATAL ERROR <"
  string_1 = "Internal error code"

  #Checking to see if there is ABNORMAL TERMINATION DUE TO FATAL ERROR
  if check_string_in_file(file_name, string_0):
    Failed_Simulations.append(Simulations[i].rsplit('/',1)[0])

#Removing the erroneous output files
for i in np.arange(len(Failed_Simulations)):
  os.chdir(Failed_Simulations[i])
  os.system('rm -r myMPI.*')
  os.system('rm -r *_output*')
  os.chdir(pfwd)

#Editing the simulation input files

files = []
for i in np.arange(len(Failed_Simulations)):
   files.append(Failed_Simulations[i]+'/simulation_input.dat')  
files.sort()

for file_to_edit_path in files:
    with open(file_to_edit_path, 'r') as file:
        # Reading the contents of the file
        lines = file.readlines()

        # Editing the desired line
    line_number_to_edit_1 = 33
    new_content_1 =  'override_array_bounds & & 200 200\n'
    lines[line_number_to_edit_1] = new_content_1

    with open(file_to_edit_path, 'w') as file:
        # Writing the modified lines back to the file
        file.writelines(lines)

#Rerunning the corresponding kMC simulation
        
kmcfiles = []
for i in np.arange(len(Failed_Simulations)):
   kmcfiles.append(glob.glob(Failed_Simulations[i]+'/*_kmc.csh'))  
#Flattening the resulting list of lists
kmcfiles = [item for sublist in kmcfiles for item in sublist]
print(kmcfiles)

cwd = os.getcwd()

for d in kmcfiles:
    os.chdir(d.rsplit('/',1)[0]) ## changing directory to where the corresponding directory is
    kmc_file = d.rsplit('/',1)[1]  ## extracting the kmc simulation file name
    os.system('sbatch ' + kmc_file)
    os.chdir(cwd)