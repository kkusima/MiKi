from main import *
import numpy as np   #package for numerical arithmetics and analysis
import matplotlib.pyplot as plt
import sys, os
# %%
# import matplotlib.pyplot as plt

# plt.figure()
# x = [1, 1]
# plt.plot(x)
# plt.show()
# import tensorflow as tf
# import torch as tf
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor



# t,covg,fits = fit.fitting_rate_param(option='ML',plot=True)
#####FITTING#------------------------------------------------------------------------------------------------------------------------------
fit = Fitting('KMC_Steady_Kinetic_Input.csv','Atomic.csv','Stoich.csv','Param.csv') #covgdep = Allowing for coverage dependance to be considered in the fit
fit.set_limits_of_integration(fit.Input.iloc[0,0],fit.Input.iloc[-1,0])
fit.n_extract = 0.5

rate_k = fit.k
act= fit.rate_func_SSKMC(5,*rate_k)
print(act)
# vec_a = fit.rate_func_0(2,*rate_k)
# print(len(vec_a))
# print(np.shape(vec_a[0]))
#print(vec_a)

#------------------------------------------------------------------------------------------------------------------------------
# MKM1.set_initial_coverages(init=[0,0,0,1]) #Sets the initial coverages of all the surface species (Note: Empty Sites are calculated Automatically. If no option is entered, default initial coverage is zero surface species coverage on the surface)
# MKM1.rate_const_correction='Forced_exp_CD'
# sol3,solt3= MKM1.solve_coverage(plot=True) #Obtains the coverages(sol) with respect to time(solt) and plots them if plot=True (Note: Additional options can be set manually - See main.py for syntax)
# sol4,solt4 = MKM1.dynamic_transient_rates_production(State1=[0.2e-9,2e-6,1e-8],State2=[0.8e-5,2e-6,1e-8],plot=True) #Calculate the transient response from State 1 to State 2. State conditions (Pressures) can be entered as seen in this line, or if not entered, a prompt will appear asking for the relevant state conditions


# plt.figure()
# sol2,solt2 = MKM1.cyclic_dynamic_transient_coverages(State1=[0.2e-9,2e-6,1e-8],State2=[0.8e-5,2e-6,1e-8],t1=2e6,t2=6e6,total_time=20e6,plot=True) #Calculate the transient response from State 1 to State 2. State conditions (Pressures) can be entered as seen in this line, or if not entered, a prompt will appear asking for the relevant state conditions
# plt.figure()
# sol2,solt2 = MKM1.cyclic_dynamic_transient_rates_production(State1=[0.2e-9,2e-6,1e-8],State2=[0.8e-5,2e-6,1e-8],t1=2e6,t2=6e6,total_time=20e6,plot=True) #Calculate the transient response from State 1 to State 2. State conditions (Pressures) can be entered as seen in this line, or if not entered, a prompt will appear asking for the relevant state conditions
# plt.ylim([-0.005e-4,0.005e-4])#for use in rate of productions


# MKM1 = MKModel('Atomic_2.csv','Stoich_2.csv','Param_2.csv') #Defining the Model
# #Intitializations
# MKM1.set_initial_coverages(init=[0,0,0]) #Sets the initial coverages of all the surface species (Note: Empty Sites are calculated Automatically. If no option is entered, default initial coverage is zero surface species coverage on the surface)
# MKM1.set_rxnconditions() #Sets the Pressures and Temperature as defined from the Param file. (Note: One can also enter them manually - See main.py for syntax)
# MKM1.set_limits_of_integration(Ti=0,Tf=6e6)#Sets the reange of time used in integration

# #Creating sample data to test out
# sol1,solt1= MKM1.solve_coverage(plot=False) #Coverages
# #Creating dataset for ML
# MKM1.create_csv(sol1,solt1,Name='coverages.csv',label='coverages') #sol1 = coverages, solt=corresponding time. These awere calculated before from MKM1


#####FITTING#------------------------------------------------------------------------------------------------------------------------------
# fit = Fitting('coverages_2_nCD.csv','Atomic_2.csv','Stoich_2.csv','Param_2_Guess.csv',CovgDep=False)
# MLPRegressor
# KNeighborsRegressor
# DecisionTreeRegressor
# RandomForestRegressor
# vec = np.array([2.0e0,6.65e-9,2.31e2,1.15e5,6.13e8,2.14e-2,2.85e-6,5.0e2])
# fit.k = vec
# t,covg,fits = fit.fitting_rate_param(option='ML',mdl='KNeighborsRegressor',maxiter=25,maxfun=1e2,plot=True,weight=1e-1)
# # print(fit.k)
# fit.k = fit.fitted_k 
# # vec = np.array([2.0e0,6.65e-9,2.31e2,1.15e5,6.13e8,2.14e-2,2.85e-6,5.0e2])
# # print(len(vec))
# # print(fit.k)
# t,covg,fits = fit.fitting_rate_param(option='ML',mdl='RandomForestRegressor',maxiter=2,maxfun=1,plot=True,weight=1e0)


# %%
