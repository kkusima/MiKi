#!/usr/bin/env python
#SBATCH -p batch
#SBATCH -o myMPI.o%j
#SBATCH -e myMPI.e%j
#SBATCH -N 1 -n 30
#SBATCH -t 5:00:00   
#SBATCH --mail-type=ALL
#SBATCH --mail-user=klkusima@uh.edu

import os 
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import shutil

Total_Pressures = np.array([1e-6,1e-3,1e-1,1e0,1e1,20])
Mol_Frac = np.array([[0,1,0],[1,0,0],[0.5,0.5,0], [0.33,0.67,0]])
no_spec_coverages_options_each = 5
#------------------Counting how many simulations combinatiuons are produced ---------------------------------
theta_A = np.linspace(0, 1, no_spec_coverages_options_each)
in_covg_count = 0 #simulation count
print('Coverages:\n')
for i in np.arange(len(theta_A)): #looping through the A species possible coverages
        if i==0:
            theta_B = theta_A
        else:
            theta_B = theta_A[:-i]
        for j in theta_B:         #looping through the corresponding remaining possible B species coverages
            cov_A = theta_A[i]
            cov_B = j            
            in_covg_count += 1
            print('Species A:',cov_A)
            print('Species B:',cov_B)
            print('----')
#----------------------------------------------------------------------------------------------------------------
print('Mol Fractions :\n O2 , CO, CO2 :\n', Mol_Frac)

print('Number of initial coverage variations per Total Pressure:',in_covg_count)
print('Number of Total Pressure variations per Mol fraction:', len(Total_Pressures))
print('Number of Mol fraction variations', len(Mol_Frac))
print('\n')
print('Overall Total number of simulations:', in_covg_count*len(Total_Pressures)*len(Mol_Frac))


#Functions to copy files and folders
def copy_file(src_path, dest_path):
    try:
        shutil.copy(src_path, dest_path)
#         print(f"File copied successfully from {src_path} to {dest_path}")
    except FileNotFoundError:
        print(f"Error: {src_path} not found.")
    except PermissionError:
        print(f"Error: Permission denied. Check if you have the necessary permissions.")
        
def copy_folder(src_path, dest_path):
    try:
        shutil.copytree(src_path, dest_path)
#         print(f"Folder copied successfully from {src_path} to {dest_path}")
    except FileNotFoundError:
        print(f"Error: {src_path} not found.")
    except PermissionError:
        print(f"Error: Permission denied. Check if you have the necessary permissions.")
    except shutil.Error as e:
        print(f"Error: {e}")

#Creating folders corresponding to Mol fraction variation and editting the simulation input file accordingly
main_folder_mol_frac = 'Mol_Fraction_variation' #Folder containing all varied initial coverage simulations
os.makedirs(main_folder_mol_frac)
mol_frac_path_list = []
for i in np.arange(np.shape(Mol_Frac)[0]):
    mol_frac_folder = ("O2_{}_CO_{}_CO2_{}".format(float(Mol_Frac[i,0]),float(Mol_Frac[i,1]),float(Mol_Frac[i,2])))
    mol_frac_path = './'+main_folder_mol_frac + '/' + mol_frac_folder
    mol_frac_path_list.append(mol_frac_path)
    
    #### Making the mol frac folders
    if not os.path.exists(mol_frac_path):
        os.makedirs(mol_frac_path)

    #### Copying the necessary input files
    source_directory = os.getcwd() 
    destination_directory = mol_frac_path
    input_folder_name = 'Other_Input_Files'
    
    source_path = os.path.join(source_directory, input_folder_name)
    destination_path = os.path.join(destination_directory, input_folder_name)
    copy_folder(source_path, destination_path)
    
    #### Editing the simulation input file to correct the mol fractions accordingly
    file_to_edit_path = os.path.join(destination_directory, input_folder_name) + '/simulation_input.dat'
        
    with open(file_to_edit_path, 'r') as file:
        # Reading the contents of the file
        lines = file.readlines()

    # Editing the desired line
    line_number_to_edit = 12  
    new_content = f"{'gas_molar_fracs'}{' ' *11}{str(Mol_Frac[i,0])}{' ' *6}{str(Mol_Frac[i,1])} {' ' *6} {str(Mol_Frac[i,2])}"+'\n'
    lines[line_number_to_edit] = new_content

    with open(file_to_edit_path, 'w') as file:
        # Writing the modified lines back to the file
        file.writelines(lines)   
    
#Creating folders corresponding to Total Pressure variation and editting the simulation input file accordingly
        
tot_pressure_folder_path_list = []
for i in np.arange(len(mol_frac_path_list)):
    for j in np.arange(len(Total_Pressures)):
        tot_pressure_folder = ("{}_bar".format(Total_Pressures[j]))
        tot_pressure_folder_path = mol_frac_path_list[i] + '/' + tot_pressure_folder
        tot_pressure_folder_path_list.append(tot_pressure_folder_path)

        #### Making the different total pressure folders
        if not os.path.exists(tot_pressure_folder_path):
            os.makedirs(tot_pressure_folder_path)
            
        #### Copying the necessary input files
        source_directory = mol_frac_path_list[i]
        destination_directory = tot_pressure_folder_path
        input_folder_name = 'Other_Input_Files'
        
        source_path = os.path.join(source_directory, input_folder_name)
        destination_path = os.path.join(destination_directory, input_folder_name)
        copy_folder(source_path, destination_path)
        
        #### Editing the simulation input file to correct the mol fractions accordingly
        file_to_edit_path = os.path.join(destination_directory, input_folder_name) + '/simulation_input.dat'

        with open(file_to_edit_path, 'r') as file:
            # Reading the contents of the file
            lines = file.readlines()

        # Editing the desired line
        line_number_to_edit = 6 
        new_content = f"{'pressure'}{' ' *18}{str(Total_Pressures[j])}"+'\n'
        lines[line_number_to_edit] = new_content

        with open(file_to_edit_path, 'w') as file:
            # Writing the modified lines back to the file
            file.writelines(lines)   

print(tot_pressure_folder_path_list)       

import time
start_time = time.time()
Simulation_path_list_i = []
for l in np.arange(len(tot_pressure_folder_path_list)):
    no_sim=5 #Number of possible coverages for each species #This increases how many total folders are made.
    Tot_no_sites = 18432 #Total Number of sites of which will be randomly filled
    A='O*'; B='CO*' #C = CO2

    theta_A = np.linspace(0, 1, no_sim) #Coverages of A
    theta_A


    main_folder = tot_pressure_folder_path_list[l]

    k=0 #Count for number of files created
    Array_tot_sites = np.arange(1,Tot_no_sites) #Array of all possible sites

    for i in np.arange(len(theta_A)): #looping through the A species possible coverages
        if i==0:
            theta_B = theta_A
        else:
            theta_B = theta_A[:-i]
        for j in theta_B:         #looping through the corresponding remaining possible B species coverages
            cov_A = theta_A[i]
            cov_B = j  
            folder_name = ("Sim_A_{}_B_{}".format(int(cov_A*100),int(cov_B*100) ))
            file_name = 'state_input.dat'
            folder_path = main_folder+'/'+folder_name
            Simulation_path_list_i.append(folder_path)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path) #Making the simulation folder:
                k += 1 #Counting how many simulation folders have been made
                if cov_A == 0 and cov_B ==0: #Accounting when you have empty surface, you dont need state input file
                    continue

                else:
                #Creating the state_input file accordingly
                    file_path = os.path.join(folder_path, file_name)
        #             print(file_path)
                    ofile=open(file_path,'w')
                    ofile.write("initial_state\n")

                    occupied_loc = [] #All site locations that are occupied

                    #For A : CO* #Choosing only x amount of active sites corresponds to A i.e based on the coverage
                    for m in np.arange(int(cov_A*Tot_no_sites)):
                        Available_loc = list(set(Array_tot_sites) - set(occupied_loc)) #List of available sites to choose from
                        if Available_loc!=[]: #Make sure there is something available
                            loc_A = random.choice(Available_loc) #Extracting a random site location from the ones available
                            occupied_loc.append(loc_A)
                            ofile.write("   seed_on_sites %s %i\n" %(A,loc_A) )     

                    #For B : O*
                    for n in np.arange(int(cov_B*Tot_no_sites)):
                        Available_loc = list(set(Array_tot_sites) - set(occupied_loc)) #List of available sites to choose from
                        if Available_loc!=[]: #Make sure there is something available
                            loc_B = random.choice(Available_loc) #Extracting a random site location from the ones available
                            occupied_loc.append(loc_B)
                            ofile.write("   seed_on_sites %s %i\n" %(B,loc_B) )                

                    ofile.write("end_initial_state") 

    print('Total number of new folders made:',k)
    count = 0
    for item in os.listdir(main_folder+'/'):
        if item.startswith('Sim'):
            count += 1
    print('Total number of simulation folders in the folder:',count)

    end_time = time.time()

elapsed_time = end_time - start_time
print("\nElapsed Coverage file generation: \n", elapsed_time, "seconds \n", elapsed_time/60, "minutes")


#Transferring the other input files into the simulation folders and editting the zacros execution file name accordingly

Simulation_number_lists = []
for i in np.arange(len(Simulation_path_list_i)):
    
    #### Calling for the source directory(corresponding to the correct pressures)
    source_directory = Simulation_path_list_i[1].rsplit('/',1)[0] + '/' + 'Other_Input_Files'

    destination_directory =  Simulation_path_list_i[i]
    input_files = ['energetics_input.dat', 'lattice_input.dat', 'mechanism_input.dat', 'simulation_input.dat']
    for k in np.arange(len(input_files)):
        input_file_name = input_files[k]

        source_path = os.path.join(source_directory, input_file_name)
        destination_path = os.path.join(destination_directory, input_file_name)
        copy_file(source_path, destination_path)

    #Renaming zacros input file to be used to track and monitor simulations
    source_path = os.path.join(source_directory, 'kmc.csh')
    Simulation_number = ("Sim_{}_kmc.csh".format(i))
    Simulation_number_lists.append(Simulation_number)
    destination_path = os.path.join(destination_directory, Simulation_number)
    copy_file(source_path, destination_path)

#Displaying the list of simulations:
Simulation_Array = np.array(list(range(0,len(Simulation_path_list_i),1)))
Simulation_path_list_i_shortened = []
for i in np.arange(len(Simulation_path_list_i)):
    Simulation_path_list_i_shortened.append(Simulation_path_list_i[i].split('/',1)[1].split('/',1)[1])
data = {'Sim_Array': Simulation_Array, 'Sim_path_shortened' : Simulation_path_list_i_shortened }
df = pd.DataFrame(data)
df