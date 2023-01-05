from main import *
import numpy as np
import matplotlib.pyplot as plt
import sys, os

MKM1 = MKModel('Atomic.csv','Stoich.csv','Param.csv') #Defining the Model
#Intitializations
MKM1.set_initial_coverages(init=[0,0,0,1]) #Sets the initial coverages of all the surface species (Note: Empty Sites are calculated Automatically. If no option is entered, default initial coverage is zero surface species coverage on the surface)
MKM1.set_rxnconditions() #Sets the Pressures and Temperature as defined from the Param file. (Note: One can also enter them manually - See main.py for syntax)
MKM1.set_limits_of_integration(Ti=0,Tf=6e6)#Sets the reange of time used in integration

#Creating sample data to test out
sol1,solt1= MKM1.solve_coverage(plot=False) #Coverages
#Creating dataset for ML
# MKM1.create_csv(sol1,solt1,Name='coverages.csv',label='coverages') #sol1 = coverages, solt=corresponding time. These awere calculated before from MKM1

print(MKM1.get_SS_X_RC_g(1))