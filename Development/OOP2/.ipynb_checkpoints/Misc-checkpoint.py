from main import *
import numpy as np
import matplotlib.pyplot as plt
import sys, os


fit = Fitting('coverages.csv','Atomic.csv','Stoich.csv','Param_Guess.csv',CovgDep=False)
# MLPRegressor
# KNeighborsRegressor
# DecisionTreeRegressor
# RandomForestRegressor
vec = np.array([2.0e0,6.65e-9,2.31e2,1.15e5,6.13e8,2.14e-2,2.85e-6,5.0e2])
fit.k = vec
t,covg,fits = fit.fitting_rate_param(option='ML',mdl='RandomForestRegressor',n=5e4,maxiter=25,maxfun=1e2,plot=True,weight=1e-1)
# print(fit.k)
fit.k = fit.fitted_k 