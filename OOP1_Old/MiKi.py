from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

Stoich = pd.read_csv("Stoic_Input.csv")    #Opening/Reading the Stoichiometric input file needed to be read
Atomic = pd.read_csv("Atom_Input.csv")     #Opening/Reading the Atomic input file needed to be read
Param = pd.read_csv("Param_Input.csv")     #Opening/Reading the Parameter input file needed to be read   

#Guess Parameters for Fiting #Initial Values for Numerical Solver
Param_Guess = pd.read_csv("Param_Input_Guess.csv")     #Opening/Reading the Parameter input file needed to be read   

#Extracting K , a, b and c coefficients 
class Coefficients:
    def __init__(self, P):
        self.P = P
        
    def Pextract(self):
        Param == self.P
        vecP=[]
        for j in np.arange(len(Param.iloc[:,0])): #looping through second column
            if ('P' in Param.iloc[j,1]):  #checking the first and second columns
                vecP.append(Param.iloc[j,2])
        return np.array(vecP) #Converts from list to array

    def kextract(self):
        Param == self.P
        veck=[]
        for j in np.arange(len(Param.iloc[:,1])): #looping through second column
            if 'k' in Param.iloc[j,1]:
                veck.append(Param.iloc[j,2])  
        return np.array(veck) #Converts from list to array

    def aextract(self):
        Param == self.P
        veca=[]
        for j in np.arange(len(Param.iloc[:,0])): #looping through second column
            if ('const' == Param.iloc[j,0]) and ('a' in Param.iloc[j,1]):  #checking the first and second columns
                veca.append(Param.iloc[j,2])
        return np.array(veca) #Converts from list to array

    def bextract(self):
        Param == self.P
        vecb=[]
        for j in np.arange(len(Param.iloc[:,0])): #looping through second column
            if ('const' == Param.iloc[j,0]) and ('b' in Param.iloc[j,1]):  #checking the first and second columns
                vecb.append(Param.iloc[j,2])
        return np.array(vecb) #Converts from list to array

    def cextract(self):
        Param == self.P
        vecc=[]
        for j in np.arange(len(Param.iloc[:,0])): #looping through second column
            if ('const' == Param.iloc[j,0]) and ('c' in Param.iloc[j,1]):  #checking the first and second columns
                vecc.append(Param.iloc[j,2])
        return np.array(vecc) #Converts from list to array

#Coverage dependancy on rate constants
def ratecoeff(kref,a,b,c,th1,th2,th3):
    K = kref*np.exp((a*th1 + b*th2 + c*th3))  #/RT lumped into a and b assuming T is constant
    return K
