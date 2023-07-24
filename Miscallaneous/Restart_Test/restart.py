#!/usr/bin/env python

#SBATCH -p batch
#SBATCH -A grabow
#SBATCH -o myMPI.o%j  
#SBATCH -e myMPI.e%j    
#SBATCH -N 1 -n 24 --exclusive
#SBATCH -t 100:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=klkusima@uh.edu        #your email id

import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np

Path = os.getcwd()
frac_O2 = np.logspace(-5,-1,3)
Simulation_directory = Path + '/Test_1/'
Starting_surface_directory = Simulation_directory + 'starting_surface/'

def replace_line(file_name, line_num, text): 
    ##This is a function to replace the line [line_num] in an input file [file_name] according to the [text] provided
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()


def create_state_input_file(file_name,n_sites):
    history_output_file = open(file_name, 'r').readlines() 
    ssp_line = history_output_file[1]
    surface_species = ssp_line[16:].split()
    surface_species
    lines = history_output_file[-(n_sites+1):][:-1] #Extracting lines corresponding to the last KMC event (added [:-1] to ignore the last line of the history output file)
    site_loc = np.arange(1,n_sites+1,1) #Creating an array of total number of sites (added n_sites+1 to account for python counting)
    lines = [i.split()for i in lines] #Splitting each line into its columns
    last_config_array = np.array([[int(float(j)) for j in i] for i in lines]) #Converting string lists of lists to int list of lists
    site_loc = last_config_array[:,0] #Extracting all possible surface sites
    species_number = last_config_array[:,2]
    #creating a dataframe of the relevant data (site location on the corresponding species number 0,1,2,...) The array needs to be transposed due to the making of an arry from two lists, the index is set to correspond to the site locations
    df_config = pd.DataFrame(np.array((site_loc,species_number)).T, index = site_loc) 
    #Removing empty sites rows so as to have a dataframe ready to be used to make the state input file
    df_config = df_config[df_config.iloc[:,1] != 0] #Only Keeping the values of whose second column doesn't have 0 (i.e an empty site)
    #loop to replace the species number with the corresponding surface species string to be easily inputted in state input
    for i in np.arange(len(df_config)):
        val = df_config.iloc[i,1] #Extract the species number
        species_replacement = surface_species[val-1]
        df_config.iloc[i,1] = species_replacement

    f = open("state_input.dat", "w+")
    f.write('initial_state')
    for i in np.unique(df_config.iloc[:,1]):
        unique_df = df_config[df_config.iloc[:,1]==i]
        for j in np.arange(len(unique_df)):
            out = unique_df.iloc[j,:] #output line to be plottes
            f.write('\n' + ' ' + ' ' + ' ' + 'seed_on_sites' + ' ' + '%s' % out[1] + ' ' + '%s' % out[0])

    f.write('\n' + 'end_initial_state' + '\n')
    f.close()


if os.path.exists(Starting_surface_directory):
    print('Directory Present')
    os.chdir(Starting_surface_directory)
    
    

history_output_file_path = Starting_surface_directory #Initializing the path where the first history output file will be used to inform the creation of the next state input file

for i in np.arange(len(frac_O2)):+
    os.system('JOBID=$(sbatch --parsable kmc.csh')
    new_directory = 'pressure_' + str(frac_O2[i]) +'/'
    os.chdir(Simulation_directory)
    os.makedirs(new_directory)
    os.chdir(Simulation_directory + new_directory)
    #Copying the non-changing input files from the starting surface directory
    os.system('cp '+ Starting_surface_directory +'/energetics_input.dat .')
    os.system('cp '+ Starting_surface_directory +'/mechanism_input.dat .')
    os.system('cp '+ Starting_surface_directory +'/lattice_input.dat .') 
    os.system('cp '+ Starting_surface_directory +'/kmc.csh .')
    #Copying the simulation input file from the strating directory and changing a line according to new simulation conditions
    os.system('cp '+ Starting_surface_directory +'/simulation_input.dat .')
    replace_line('simulation_input.dat',12,'gas_molar_fracs           %s' % frac_O2[i] + '      %s' % 0.00 + '      %s' % 0.00 + '\n')

    #Extracting number of totals sites according to general_output file from the starting surface directory (Note this doesn't change since the surface size remains the same)
    inp=open(Starting_surface_directory +'general_output.txt','r').readlines()
    for i in np.arange(len(inp)):
        if 'Total number of lattice sites:' in inp[i]:
            val = i  #Line in text file where sentence is present.                       
    n_sites = int(inp[val][35:]) #35 corresponds to the characters present before the number is outputted

    #Setting the history output file path 
    history_output =  history_output_file_path + 'history_output.txt'

    create_state_input_file(history_output,n_sites) #Using the function to create the state input file from the previous run's history output
    #Updating the history output file path to correspond to the previous directory so that the correct history can be passed down to make the new state input for the next run
    history_output_file_path = Simulation_directory + new_directory
    os.system('sbatch --dependency after:$JOBID a_file.sh')