#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:55:52 2022

@author: klkusima
"""

from main import *
import numpy as np
import matplotlib.pyplot as plt
import sys, os


fit = Fitting('coverages.csv','Atomic.csv','Stoich.csv','Param_Guess.csv',CovgDep=False)
# MLPRegressor
# KNeighborsRegressor
# DecisionTreeRegressor
# RandomForestRegressor
t,covg,fits = fit.fitting_rate_param(option='min',method_min='nelder-mead',mdl='RandomForestRegressor',n=5e4,maxiter=1e5,maxfun=1e5,plot=True,weight=1e-1)
# print(fit.k)
fit.k = fit.fitted_k 
# sol,solt = fit.solve_coverage(plot=True)



