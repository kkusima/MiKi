from main import *
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# import tensorflow as tf
# import torch as tf
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor



# t,covg,fits = fit.fitting_rate_param(option='ML',plot=True)
#####CYCLIC OP#------------------------------------------------------------------------------------------------------------------------------

# MKM1 = MKModel_wCD('Atomic.csv','Stoich.csv','Param.csv') #Defining the Model
# MKM1.set_initial_coverages(init=[0,0,0]) #Sets the initial coverages of all the surface species (Note: Empty Sites are calculated Automatically. If no option is entered, default initial coverage is zero surface species coverage on the surface)
# MKM1.set_rxnconditions() #Sets the Pressures and Temperature as defined from the Param file. (Note: One can also enter them manually - See main.py for syntax)
# MKM1.set_limits_of_integration(Ti=0,Tf=6e6)#Sets the reange of time used in integration

# plt.figure()
# sol2,solt2 = MKM1.cyclic_dynamic_transient_coverages(State1=[0.2e-9,2e-6,1e-8],State2=[0.8e-5,2e-6,1e-8],t1=2e6,t2=6e6,total_time=20e6,plot=True) #Calculate the transient response from State 1 to State 2. State conditions (Pressures) can be entered as seen in this line, or if not entered, a prompt will appear asking for the relevant state conditions
# plt.figure()
# sol2,solt2 = MKM1.cyclic_dynamic_transient_rates_production(State1=[0.2e-9,2e-6,1e-8],State2=[0.8e-5,2e-6,1e-8],t1=2e6,t2=6e6,total_time=20e6,plot=True) #Calculate the transient response from State 1 to State 2. State conditions (Pressures) can be entered as seen in this line, or if not entered, a prompt will appear asking for the relevant state conditions
# plt.ylim([-0.005e-4,0.005e-4])#for use in rate of productions

#####FITTING#------------------------------------------------------------------------------------------------------------------------------
fit = Fitting('coverages.csv','Atomic.csv','Stoich.csv','Param_Guess.csv',CovgDep=False)

fit.ML_data_gen()
# t,covg,fits = fit.fitting_rate_param(option='ML',mdl='RandomForestRegressor',plot=True)
