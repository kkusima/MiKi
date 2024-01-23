import pandas as pd
import os
import numpy as np
import random

#Request the initial coverage of A : O* and B : CO*
cov_A = float(input("Initial Coverage of O* (A): "))
cov_B = float(input("Initial Coverage of CO* (B): "))
#Request Total number of Sites
Tot_no_sites = int(input("Total Number of Sites: "))#18432 #Total Number of sites of which will be randomly filled

A = 'O*'
B = 'CO*'
Array_tot_sites = np.arange(1,Tot_no_sites) #Array of all possible sites
file_name = 'state_input.dat'

ofile=open(file_name,'w')
ofile.write("initial_state\n")

occupied_loc = [] #All site locations that are occupied

#For A : O* #Choosing only x amount of active sites corresponds to A i.e based on the coverage
for m in np.arange(int(cov_A*Tot_no_sites)):
    Available_loc = list(set(Array_tot_sites) - set(occupied_loc)) #List of available sites to choose from
    if Available_loc!=[]: #Make sure there is something available
        loc_A = random.choice(Available_loc) #Extracting a random site location from the ones available
        occupied_loc.append(loc_A)
        ofile.write("   seed_on_sites %s %i\n" %(A,loc_A) )     

#For B : CO*
for n in np.arange(int(cov_B*Tot_no_sites)):
    Available_loc = list(set(Array_tot_sites) - set(occupied_loc)) #List of available sites to choose from
    if Available_loc!=[]: #Make sure there is something available
        loc_B = random.choice(Available_loc) #Extracting a random site location from the ones available
        occupied_loc.append(loc_B)
        ofile.write("   seed_on_sites %s %i\n" %(B,loc_B) )                

ofile.write("end_initial_state") 
        
print('state_input.dat file has been created!')
