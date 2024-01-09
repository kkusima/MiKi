import glob
import re
import os
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

Failed_KMC_Simulations = glob.glob('Mol_Fraction_variation/O2*/*_bar/Sim_*/*_kmc*')
Simulations = Failed_KMC_Simulations
Simulation_number = []
for i in np.arange(len(Simulations)):
    Simulation_number.append(int(Simulations[i].rsplit('/',1)[1].rsplit('_',1)[0].rsplit('_',1)[1]))
Simulation_number = [int(x) for x in Simulation_number]

#Performing Human sorting
def natural_sort_key(item):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', item[0])]

a = np.array([Simulation_number,Simulations]).T
data = a

sorted_data = np.array(sorted(data, key=natural_sort_key))

import csv

# Creating the two lists
list1 = sorted_data[:,0]
list2 = sorted_data[:,1]

# Openning the CSV file in write mode
with open('Simulation_files_list.csv', 'w', newline='') as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)

    # Iterate over the lists and write each element to a separate column
    for i in range(len(list1)):
        writer.writerow([list1[i], list2[i]])