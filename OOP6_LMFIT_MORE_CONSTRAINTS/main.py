import sys, os
import numpy as np   #package for numerical arithmetics and analysis
import pandas as pd  #package for dataframe and file extraction/creation
import string        #package to allow for access to alphabet strings
import math          #package to allow for the use of mathematical operators like permutation calculation
from mpmath import * #package for precision control
dplace=10    #Controls decimal places - used for mp.dps in mpmath precision control
import matplotlib.pyplot as plt         #package for plotting
from scipy.integrate import solve_ivp   #ODE solver
from scipy import optimize
from scipy.special import logsumexp
from numdifftools import Jacobian, Hessian
from autograd import jacobian, hessian
import lmfit
from lmfit import Model, Parameters,minimize, fit_report
import copy

# Disable Printing
def blockPrint():
    sys.__stdout__ = sys.stdout
    sys.stdout = open(os.devnull, 'w')

# Restore Printing
def enablePrint():
    sys.stdout = sys.__stdout__
    
#------------------------------------------------------------------------------------------------------------------------------    
class MKModel:
    def __init__(self,Atomic_csv,Stoich_csv,Param_csv): #Inputs necessary to initialize the MK Model
        self.Atomic = pd.read_csv(Atomic_csv)     #Opening/Reading the Atomic input file needed to be read
        self.Stoich = pd.read_csv(Stoich_csv)    #Opening/Reading the Stoichiometric input file needed to be read
        self.Param = pd.read_csv(Param_csv)     #Opening/Reading the Parameter input file needed to be read         
        
        self.check_massbalance(self.Atomic,self.Stoich) #Uses the stoich and atomic matrices to check that mass is conserved A*v=0
        
        self.k = self.kextract()    #Extracting the rate constants from the Param File (Note that format of the Param File is crucial)
        self.P,self.Temp = self.set_rxnconditions() #Setting reaction conditions (defaulted to values from the Param File but can also be set mannually )
        self.rate_const_correction = 'None' #Accounting for correction to the rate constants (i.e. enhancing the mean field approximation)
        self.BG_matrix='auto' #Bragg williams constant matrix
        self.Coeff = self.Coeff_extract() #Extracting the coverage dependance coefficients
        self.Ti,self.Tf=self.set_limits_of_integration() #Sets the range of time needed to solve for the relavant MK ODEs, defaults to 0-6e6 but can also be manually set
        self.init_cov=self.set_initial_coverages() #Sets the initial coverage of the surface species, defaults to zero coverage but can also be set manually
        
        self.status='Waiting' #Used to observe the status of the ODE Convergence
        self.label='None'   #Used to pass in a label so as to know what kind of figure to plot    
    #------------------------------------------------------------------------------------------------------------------------------    
    def check_massbalance(self,Atomic,Stoich): #Function to check if mass is balanced
        at_mat = Atomic.iloc[0:,1:]           #The atomic matrix
        err = 0                               #For counting error
        for i in np.arange(len(Stoich)):    
            st_mat = Stoich.iloc[i,1:]        #The stoichiometric matrix
            res = np.dot(at_mat,st_mat)       #Performing the matrix product for every reaction i
            if any(a != 0 for a in res):      #Verifies that the matrix product returns 0s (i.e mass is balanced)
                text = "Mass is not conserved in reaction %i. \n ... Check and correct the Atomic and/or Stoichiometric Matrices"%(i+1)
                err +=1
                raise Exception(text,'\n')
            elif (i == len(Stoich)-1 and err==0):
                text = "Mass is conserved."
                return print(text,'\n')
    #------------------------------------------------------------------------------------------------------------------------------        
    def check_coverages(self,vec):  #Function to check if the coverages being inputted make sense (Note in this code empty sites are not inputted, they're calculated automatically)
        if (np.round(float(np.sum(vec)),0))!=1 or (all(x >= 0 for x in vec)!=True) or (all(x <= 1 for x in vec)!=True):
            raise Exception('Error: The initial coverages entered are not valid. Issues may include:'
                            '\n 1. Sum of initial coverages enetered does not add up to 1 ; '
                            '\n 2. Initial coverages enetered has a number X>1 or X<0 ;'
                            '\n Please double check the initial coverages entered and make the necessary corrections')
        else:
            return vec
    #------------------------------------------------------------------------------------------------------------------------------    
    def Pextract(self): #Function used for extracting pressures from the Param File
        vecP=[]
        for j in np.arange(len(self.Param.iloc[:,0])): #looping through second column
            if ('P' in self.Param.iloc[j,1]):  #checking the first and second columns
                vecP.append(self.Param.iloc[j,2])
        return np.array(vecP) #Converts from list to array
    #------------------------------------------------------------------------------------------------------------------------------
    def kextract(self): #Function used for extracting rate constants from the Param File
        veck=[]
        for j in np.arange(len(self.Param.iloc[:,1])): #looping through second column
            if 'k' in self.Param.iloc[j,1]:
                veck.append(self.Param.iloc[j,2])  
        return np.array(veck) #Converts from list to array
    #------------------------------------------------------------------------------------------------------------------------------
    def Coeff_extract(self):
       colmn = len(self.Stoich.iloc[0,1:]) - len(self.P) #Number of columns (i.e rate coefficients = no. of surface species being investigated)
       row = len(self.k) #number of rows = number of rate constants (i.e reaction steps)
       Coeff = np.empty([row,colmn]) #initializing the coefficient matrix
       index = list(string.ascii_lowercase)[:colmn] #index holding the relevant letters (a-w) (Therefore 23 different species only possible)
       
       #Counting how many 'const' are in the Param file tobe used to check whether to use param file
       ls = self.Param.values.tolist() #The Param File as a list
       lscount = 0
       for i in np.arange(np.shape(ls)[0]):
           if ('const' in ls[i]):
               lscount = lscount + 1
       
       if self.BG_matrix == 'auto':
           if lscount==row*(colmn-1): #Param file must have exact number of constants (even if 0 has to be used) or else a 1 matrix is assumed #colm - 1 is done to remove empty coverage
           
           #Extracting the coefficients from the Param matrix
               for i in np.arange(colmn):
                   count = 0 #Corresponds to reaction. Also, treats forward and revers rxn separately
                   for j in np.arange(len(self.Param.iloc[:,0])): #looping through all param file rows
                       if ('const' == self.Param.iloc[j,0]) and (str(index[i]) in self.Param.iloc[j,1]):
                           Coeff[count][i]=self.Param.iloc[j,2]
                           count += 1            
           else:
                print("Note: Constant coefficients aren't specified or don't match model requirements.\n A default uniform matrix coefficient of ones has been used.")                    
                Coeff = np.ones((row,(colmn-1)))
                
        # elif self.BG_matrix == 'manual': #####################  Needs to be updated to allow for external manual inputting of BG_matrix  ###########################
       
       return Coeff
   #------------------------------------------------------------------------------------------------------------------------------
    def set_initial_coverages(self,init=[]): #empty sites included at the end of the code 
        mp.dps= dplace
        
        ExpNoCovg = len(self.Stoich.iloc[0,len(self.P)+1:])
        if init==[]: 
            zeros=np.zeros(ExpNoCovg-1)
            empty_sites = 1 - np.sum(zeros)
            init = np.append(zeros,empty_sites)
            
        if len(init)!=(ExpNoCovg):
            raise Exception('Number of coverage entries do not match what is required. %i entries are needed. (Remember to also include the number/coverage of empty sites).'%(ExpNoCovg))
        else: 
            #Changing the number of decimal places/precision of input coverages
            for i in np.arange(len(init)):
                init[i]=mpf(init[i])
                
            self.init_cov = init
                
        return self.check_coverages(self.init_cov)
    #------------------------------------------------------------------------------------------------------------------------------    
    def set_rxnconditions(self,Pr=None,Temp=None): #Function used for setting the reaction Pressure and Temperature (Currently Temperature is not in use)
        if Temp==None:
            Temp = self.Param.iloc[0,2]
        
        ExpNoP=len(np.array(self.Pextract()))
        
        if Pr==None: 
            self.P = np.array(self.Pextract())
        else: 
            if len(Pr)!=ExpNoP:
                raise Exception('Number of pressure entries do not match what is required. %i entries are needed.'%(ExpNoP))
            else:
                self.P = Pr    
        self.Temp = Temp
        
        return self.P,self.Temp
    #------------------------------------------------------------------------------------------------------------------------------
    def set_limits_of_integration(self,Ti=0,Tf=6e6): #Function used for setting the time limits of integration 
        self.Ti=Ti
        self.Tf=Tf
        return self.Ti,self.Tf
    #------------------------------------------------------------------------------------------------------------------------------
    def ratecoeff(self,kref,Coeff,Theta):
        if self.rate_const_correction=='None':
            K = kref
            return K
        elif self.rate_const_correction=='Forced_exp_CD': #Forced exponential coverage dependance
            if len(Coeff) != len(Theta):
                raise Exception('The number of the coefficients doesnt match the relevant coverages. Please make sure to check the Parameters csv file for any errors. ')
            else:
                K = kref*np.exp(float(logsumexp(np.sum(np.multiply(Coeff,Theta)))))  #/RT lumped into a and b assuming T is constant
                return K
    #------------------------------------------------------------------------------------------------------------------------------
    def get_rates(self,cov=[]): #cov = coverages  #Function used to calculate the rates of reactions
        
        if cov==[]:
            cov=self.init_cov
            
        THETA = cov #Coverages being investigated

        Nr = len(self.Stoich) #Number of rows in your your stoich matrix, i.e (Number of reactions)
       

        kf = self.k[0::2] #Pulling out the forward rxn rate constants (::2 means every other value, skip by a step of 2)
        kr = self.k[1::2] #Pulling out the reverse rxn rate constants 
        
        ccolmn = len(self.Stoich.iloc[0,1:]) - len(self.P) #No. of columns in default coefficient matrix
        crow = len(self.k) #No. of rows in default coefficient matrix. Note, forward and reverse rxns are separate rxns
            
        Coeff_f = self.Coeff[0::2] #Pulling out the forward coefficients (::2 means every other value, skip by a step of 2)
        Coeff_r = self.Coeff[1::2] #Pulling out the reverse coefficients

        r = [None] * Nr  #Empty Vector for holding rate of a specific reaction
        
        #Calculating the rates of reactions:
        for j in np.arange(Nr):   #Looping through the reactions
            matr = np.concatenate((self.P,THETA),axis=0)  #concatenating into the matrix, matr
            Ns = len(matr) #Number of species *****
            fwd = []
            rvs = []
                
            for i in np.arange(Ns):
                if self.Stoich.iloc[j,i+1]<0: #extracting only forward relevant rate parameters  #forward rxn reactants /encounter probability
                    fwd.append(matr[i]**abs(self.Stoich.iloc[j,i+1]))
                    
                elif self.Stoich.iloc[j,i+1]>0: #extracting only reverse relevant rate parameters  #reverse rxn reactants /encounter probability
                    rvs.append(matr[i]**abs(self.Stoich.iloc[j,i+1]))   
                    
            r[j] = (self.ratecoeff(kf[j],Coeff_f[j][:],THETA[:])*np.prod(fwd)) - (self.ratecoeff(kr[j],Coeff_r[j][:],THETA[:])*np.prod(rvs)) #Calculating the rate of reaction
         
        r = np.transpose(r)
        
        return r  
    #------------------------------------------------------------------------------------------------------------------------------
    def get_ODEs(self,t,cov,coverage=True): #t only placed for solve_ivp purposes #Functions used for calculating the rates of production
                            #coverage flag True if excluding the rate of production of gas species
                            
        r = self.get_rates(cov)
    
        Nr = len(self.Stoich) #Number of rows in your your stoich matrix, i.e (Number of reactions)
        Ns = len(self.Stoich.iloc[0,1:]) #Number of species *****
        D = []      #Empty Vector For holding rate of change of coverage values
        #Differential Equations to calculate the rate of change in coverages
        for i in np.arange(Ns):
            dsum=0
            for j in np.arange(Nr):
                
                dsum = dsum + float(self.Stoich.iloc[j,i+1])*r[j] #Calculating the rate of production of a species i
            
            D.append(dsum)    
        
        D = np.transpose(D)    
        if coverage==True:    
            return D[len(self.P):]
        else:
            return D
    #------------------------------------------------------------------------------------------------------------------------------      
    def solve_coverage(self,t=[],initial_cov=[],method='BDF',reltol=1e-8,abstol=1e-8,Tf_eval=[],full_output=False,plot=False): #Function used for calculating (and plotting) single state transient coverages
        #Function used for solving the resulting ODEs and obtaining the corresponding surface coverages as a function of time
        if t==[]:  #Condition to make sure default time is what was set initially (from self.set_limits_of_integration()) and if a different time range is entered, it will be set as the default time limits of integration
            t=[self.Ti,self.Tf]  
        else:
            self.set_limits_of_integration(t[0],t[1])

        t_span = (t[0],t[1]) #Necessary for ODE Solver
        
        if initial_cov==[]: #Condition to make sure default initial condition is what was set initially (from self.set_initial_coverages()) and if  different initial coverages are entered, they will be set as the default intial ccoverages
            init = self.init_cov
        else:
            init = self.set_initial_coverages(initial_cov)
                        
        #Necessary to allow Teval to be set by simply entering Tf_eval which would correspond to the end of the range
        if Tf_eval==[]:
            T_eval=None
        else:
            T_eval=Tf_eval
            
        solve = solve_ivp(self.get_ODEs,t_span,init,method,t_eval=T_eval,rtol=reltol,atol=abstol,dense_output=full_output) #ODE Solver
        
        #COnvergence Check
        if solve.status!=0:
            self.status = 'Convergence Failed'
            raise Exception('ODE Solver did not successfuly converge. Please check model or tolerances used')
        elif solve.status==0:
            self.status = 'ODE Solver Converged'
            # print(self.status)
        
        #Extracting the Solutions:
        sol = np.transpose(solve.y)
        solt = np.transpose(solve.t)
        
        self.label='coverages'
        
        # Plotting
        
        if plot==False:
            return sol,solt
        elif plot==True:
            self.plotting(sol,solt,self.label)
            return sol,solt
    #------------------------------------------------------------------------------------------------------------------------------
    def solve_rate_reaction(self,tf=None,Tf_eval=[],initial_coverage=[],plot=False): #Function used for calculating (and plotting) single state transient rates of reaction
        
        if tf==None: 
            tf=self.Tf
        #Necessary to allow Teval to be set by simply entering Tf_eval which would correspond to the end of the range
        if Tf_eval==[]:
            T_eval=None
        else:
            T_eval=Tf_eval
        
        covg,covgt =self.solve_coverage(t=[self.Ti,tf],initial_cov=initial_coverage,Tf_eval=T_eval)
        rates_r = []
        for t in np.arange(len(covgt)):
            rates_r.append(self.get_rates(cov = covg[t,:]))
                        
        rates_r = np.array(rates_r)
        
        self.label='rates_p'
        if plot==False:
            return rates_r,covgt
        elif plot==True:
            self.plotting(rates_r,covgt,self.label)
            return rates_r,covgt
    #------------------------------------------------------------------------------------------------------------------------------
    def solve_rate_production(self,tf=None,Tf_eval=[],initial_coverage=[],plot=False): #Function used for calculating (and plotting) single state transient rates of production
        
        if tf==None:
            tf=self.Tf
        
        #Necessary to allow Teval to be set by simply entering Tf_eval which would correspond to the end of the range
        if Tf_eval==[]:
            T_eval=None
        else:
            T_eval=Tf_eval

        covg,covgt =self.solve_coverage(t=[self.Ti,tf],initial_cov=initial_coverage,Tf_eval=T_eval)
        rates_p = []
        for t in np.arange(len(covgt)):
            rates_p.append(self.get_ODEs(covgt[t],covg[t,:],coverage=False))
                        
        rates_p = np.array(rates_p)
        
        self.label='rates_p'
        if plot==False:
            return rates_p,covgt
        elif plot==True:
            self.plotting(rates_p,covgt,self.label)
            return rates_p,covgt
    #------------------------------------------------------------------------------------------------------------------------------ 
    #Functions neccessary for calculating the steady state values (Needed for when attempting dynamic switching between pressures)
    #------------------------------------------------------------------------------------------------------------------------------    
    def check_SS(self,trans_vec,tol=0.10,feature=None): #Function for checking if steady state has been reached
        #trans_vector=transient vector #tol=tolerance value i.e what percent distance is between end and end_prev (0.1 means that end_prev is a value 10% away from end)
        length = np.shape(trans_vec)[0]
        end = trans_vec[-1,:]
        end_prev = trans_vec[-int(np.round(length*tol)),:]
        steady_diff = np.abs(end-end_prev)
        
        msg='Steady State Reached'
        if feature=='coverage': 
            if all(x < 1e-2 for x in steady_diff):
                return (end,msg)
            else:
                msg = 'Warning: STEADY STATE MAY NOT HAVE BEEN REACHED. Difference in a set of last two coverage terms is NOT less than 1e-2.Last terms are returned anyways.'
                return (end,msg)
        elif feature=='rates_reaction': 
            if all(x < 1e-7 for x in steady_diff):
                return (end,msg)
            else:
                msg = 'Warning: STEADY STATE MAY NOT HAVE BEEN REACHED. Difference in a set of last two rates of reaction terms is NOT less than 1e-7. Last terms are returned anyways.'
                return (end,msg)
        elif feature=='rates_production': 
            if all(x < 1e-7 for x in steady_diff):
                return (end,msg)
            else:
                msg = 'Warning: STEADY STATE MAY NOT HAVE BEEN REACHED. Difference in a set of last two rates of production terms is NOT less than 1e-7. Last terms are returned anyways.'
                return (end,msg)
    #------------------------------------------------------------------------------------------------------------------------------
    def get_SS_coverages(self,tf=None,Tf_eval=[]): #Function used for calculating the steady state coverages
        if tf==None:
            tf=self.Tf

        covg,covgt = self.solve_coverage(t=[self.Ti,tf],Tf_eval=Tf_eval)
        
        SS,msg = self.check_SS(covg,feature='coverage')
        print(msg)
        return SS
    #------------------------------------------------------------------------------------------------------------------------------    
    def get_SS_rates_reaction(self,tf=None,Tf_eval=[]): #Function used for calculating the steady state rates of reaction
        rates_r,time_r = self.solve_rate_reaction(tf=tf,Tf_eval=Tf_eval)
        
        SS,msg = self.check_SS(rates_r,feature='rates_reaction')
        print(msg)
        return SS
    #------------------------------------------------------------------------------------------------------------------------------
    def get_SS_rates_production(self,tf=None,Tf_eval=[]): #Function used for calculating the steady state rates of production
        rates_p,time_R = self.solve_rate_production(tf=tf,Tf_eval=Tf_eval)  
        
        SS,msg = self.check_SS(rates_p,feature='rates_production')
        print(msg)
        return SS
    
    #------------------------------------------------------------------------------------------------------------------------------
    def get_X_RC_SS(self,p_inc=0.1,k_o_inp=[],rxn = -1):
        #p_inc is the percent increase of rate const. #k_o_inp, is the inputed rate constants to allow for their initial values to be user defined, #rxn is the reaction producing the products being investigated : -1 indicates the last elementary step
        if k_o_inp!=[]:
            k_o = k_o_inp
        else:
            k_o = self.kextract() #From Param file
            
        Xrc = [] #Initializing the empty array of degrees of rate control
        rin = self.get_SS_rates_reaction()
        
        if rxn>len(rin) or rxn<(-len(rin)):
            raise Exception('An invalid rxn value has been entered')
        else:
            ro = rin[rxn] 
        
        for i in np.arange(len(rin)):
            n = 2*i
            # knew = k_o #Re-initializing knew so it can only be changed for the specific elementary steps
            enablePrint()
            print(i)
            print('before:')
            print(k_o)
            kfwd = k_o[n]*(1+p_inc)
            krvs = k_o[n+1]*(1+p_inc)
            indices = [n,n+1]
            repl = [kfwd,krvs]
            knew = k_o[:] 
            for index, replacement in zip(indices, repl):
                knew[index] = replacement
            print('after')
            print(knew)
            self.k = knew
            knew = k_o
            rnew =self.get_SS_rates_reaction()
            Xrc.append((rnew[i]-ro)/(ro*p_inc))
            blockPrint()
                
        return Xrc
    
    #------------------------------------------------------------------------------------------------------------------------------
    #Functions to calculate the resulting kinteic parametes from pressure switching
    #------------------------------------------------------------------------------------------------------------------------------
    def Dynamic(self,State1=[],State2=[],t1=None): #Function used for storing and prompting the input of the two states involved when dynamically switching pressures
        if State1==[]:
            print('\nThe Pressure Input Format:P1,P2,P3,...\n')
            print("Enter the Pressure Conditions of State 1 below:")
            State1_string = input().split(',')
            State1 = [float(x) for x in State1_string]
        if State2==[]:  
            print("Enter the Pressure Conditions of State 2 below:")
            State2_string = input().split(',')
            State2 = [float(x) for x in State2_string]
            
        if t1!=None:
            self.Tf = t1
            blockPrint() #Disable printing (since warning will show up if steady state not reached)
            self.set_rxnconditions(Pr=State1)
            SS_State1 = self.get_SS_coverages()
            enablePrint() #Re-enable printing
        else:
            self.set_rxnconditions(Pr=State1)
            SS_State1 = self.get_SS_coverages()

        return SS_State1,State2
    #------------------------------------------------------------------------------------------------------------------------------
    def dynamic_transient_coverages(self,State1=[],State2=[],t1=None,t2=None,plot=False): #Function used for calculating (and plotting) the dynamic transient coverages
        
        SS_State1,State2 = self.Dynamic(State1,State2,t1)
        
        if t2!=None and t1!=None:
            self.Ti =t1
            self.Tf = t1+t2
            self.set_rxnconditions(Pr=State2)
            sol,solt = self.solve_coverage(initial_cov=SS_State1)
        elif (t2!=None and t1==None) or (t1!=None and t2==None):
            raise Exception("Either both t1 and t2 should be inputted or neither.")
        else:
            self.set_rxnconditions(Pr=State2)
            sol,solt = self.solve_coverage(initial_cov=SS_State1)
            
        self.label='coverages'
        if plot==False:
            return sol,solt
        elif plot==True:
            self.plotting(sol,solt,self.label)
            return sol,solt
    #------------------------------------------------------------------------------------------------------------------------------
    def dynamic_transient_rates_reaction(self,State1=[],State2=[],t1=None,t2=None,plot=False): #Function used for calculating (and plotting) the dynamic transient rates of reaction
        
        SS_State1,State2 = self.Dynamic(State1,State2,t1)
        
        if t2!=None and t1!=None:
            self.Ti =t1
            self.Tf = t1+t2
            self.set_rxnconditions(Pr=State2)
            sol,solt = self.solve_rate_reaction(initial_coverage=SS_State1)
        elif (t2!=None and t1==None) or (t1!=None and t2==None):
            raise Exception("Either both t1 and t2 should be inputted or neither.")
        else:
            self.set_rxnconditions(Pr=State2)
            sol,solt = self.solve_rate_reaction(initial_coverage=SS_State1)

        self.label='rates_r'
        if plot==False:
            return sol,solt
        elif plot==True:
            self.plotting(sol,solt,self.label)
            return sol,solt
    #------------------------------------------------------------------------------------------------------------------------------
    def dynamic_transient_rates_production(self,State1=[],State2=[],t1=None,t2=None,plot=False): #Function used for calculating (and plotting) the dynamic transient rates of production
        
        SS_State1,State2 = self.Dynamic(State1,State2,t1)
        
        if t2!=None and t1!=None:
            self.Ti =t1
            self.Tf = t1+t2
            self.set_rxnconditions(Pr=State2)
            sol,solt = self.solve_rate_production(initial_coverage=SS_State1)
        elif (t2!=None and t1==None) or (t1!=None and t2==None):
            raise Exception("Either both t1 and t2 should be inputted or neither.")
        else:
            self.set_rxnconditions(Pr=State2)
            sol,solt = self.solve_rate_production(initial_coverage=SS_State1)

        self.label='rates_p'
        if plot==False:
            return sol,solt
        elif plot==True:
            self.plotting(sol,solt,self.label)
            return sol,solt
    #------------------------------------------------------------------------------------------------------------------------------
    #Functions to calculate the resulting kinteic parametes from pressure switching
    #------------------------------------------------------------------------------------------------------------------------------
    def cyclic_dynamic_transient_coverages(self,State1=[],State2=[],t1=None,t2=None,total_time=None,plot=False): #Function used for calculating (and plotting) the dynamic transient coverages
        if total_time!=None:
            total_time_f=0 #final full simulation time
            self.set_rxnconditions(Pr=State1)
            sola,solta=self.solve_coverage(t=[0,t1])#calculating first pulse (obtaining/initialising first data set for full simulation)
            full_covg =sola #Initializing full vector of rates_r
            full_time=solta #Intiialising full vector of times

            while (total_time_f<total_time):

                blockPrint() #Disable printing (since warning will show up if steady state not reached)
                solb,soltb=self.dynamic_transient_coverages(State1,State2,t1,t2)
                soltb = soltb-soltb[0]
                
                full_covg=np.vstack([full_covg,solb])
                full_time=np.hstack([full_time,full_time[-1]+soltb])
                
                total_time_f=full_time[-1]
                enablePrint() #re-enabling printing
                
            self.label='coverages'
            if plot==False:
                return full_covg,full_time
            elif plot==True:
                self.plotting(full_covg,full_time,self.label)
                return full_covg,full_time
    #------------------------------------------------------------------------------------------------------------------------------        
    def cyclic_dynamic_transient_rates_reaction(self,State1=[],State2=[],t1=None,t2=None,total_time=None,plot=False): #Function used for calculating (and plotting) the dynamic transient coverages
        if total_time!=None:
            total_time_f=0 #final full simulation time
            self.set_rxnconditions(Pr=State1)
            sola,solta=self.solve_rate_reaction(tf=t1)#calculating first pulse (obtaining/initialising first data set for full simulation)
            full_rt_r =sola #Initializing full vector of rates_p
            full_time=solta #Intiialising full vector of times

            while (total_time_f<total_time):

                blockPrint() #Disable printing (since warning will show up if steady state not reached)
                solb,soltb=self.dynamic_transient_rates_reaction(State1,State2,t1,t2)
                soltb = soltb-soltb[0]
                
                full_rt_r=np.vstack([full_rt_r,solb])
                full_time=np.hstack([full_time,full_time[-1]+soltb])
                
                total_time_f=full_time[-1]
                enablePrint() #re-enabling printing
                
            self.label='rates_r'
            if plot==False:
                return full_rt_r,full_time
            elif plot==True:
                self.plotting(full_rt_r,full_time,self.label)
                return full_rt_r,full_time
    #------------------------------------------------------------------------------------------------------------------------------        
    def cyclic_dynamic_transient_rates_production(self,State1=[],State2=[],t1=None,t2=None,total_time=None,plot=False): #Function used for calculating (and plotting) the dynamic transient coverages
        if total_time!=None:
            total_time_f=0 #final full simulation time
            self.set_rxnconditions(Pr=State1)
            sola,solta=self.solve_rate_production(tf=t1)#calculating first pulse (obtaining/initialising first data set for full simulation)
            full_rt_p =sola #Initializing full vector of coverages
            full_time=solta #Intiialising full vector of times

            while (total_time_f<total_time):

                blockPrint() #Disable printing (since warning will show up if steady state not reached)
                solb,soltb=self.dynamic_transient_rates_production(State1,State2,t1,t2)
                soltb = soltb-soltb[0]
                
                full_rt_p=np.vstack([full_rt_p,solb])
                full_time=np.hstack([full_time,full_time[-1]+soltb])
                
                total_time_f=full_time[-1]
                enablePrint() #re-enabling printing
                
            self.label='rates_p'
            if plot==False:
                return full_rt_p,full_time
            elif plot==True:
                self.plotting(full_rt_p,full_time,self.label)
                return full_rt_p,full_time
    #------------------------------------------------------------------------------------------------------------------------------
    def create_csv(self,sol=[],solt=[],k_inp=[],Name=None,label=None):
        
        if label==None:
            label=self.label  #Using the most recent label
        
        if label not in ('coverages','rates_r','rates_p','rate_coeff'): #Making sure one of the labels is chosen (i.e not None)
            raise Exception("The entered label is incorrect. Please insert either 'coverage' or 'rates_r' or 'rates_p' or 'rate_coeff' ")
        
        if Name!=None and Name[-4:]!='.csv':  #Making sure that the Name entered has .csv attachment
            raise Exception("Name entered must end with .csv ; Example coverages.csv")
            
        if (sol!=[] and solt!=[]):
            dat = np.insert(sol,0,solt,axis=1)   #Merging time and parameters
            dat_df = pd.DataFrame(dat)           #Creating a datframe 
            length_entry = len(dat_df.iloc[0,:])-1  #Length of the dataframe columns
        
        if label=='coverages':
            if (length_entry != len(self.Atomic.columns.values[1+len(self.P):])) :
                raise Exception('Labels dont match size of input')
            dat_df.columns = np.insert(self.Atomic.columns.values[1+len(self.P):],0,'time')
            if Name==None:
                dat_df.to_csv(label+'.csv', encoding='utf-8', index=False)
            else:
                dat_df.to_csv(Name, encoding='utf-8', index=False)
                
        elif label=='rates_r':
            if (length_entry != len(self.Stoich.iloc[:,0])) :
                raise Exception('Labels dont match size of input')
            dat_df.columns = np.insert(np.array(self.Stoich.iloc[:,0]),0,'time')
            if Name==None:
                dat_df.to_csv(label+'.csv', encoding='utf-8', index=False)
            else:
                dat_df.to_csv(Name, encoding='utf-8', index=False)
        
        elif label=='rates_p':
            if (length_entry != len(self.Atomic.columns.values[1:])) :
                raise Exception('Labels dont match size of input')
            dat_df.columns = np.insert(self.Atomic.columns.values[1:],0,'time')
            if Name==None:
                dat_df.to_csv(label+'.csv', encoding='utf-8', index=False)
            else:
                dat_df.to_csv(Name, encoding='utf-8', index=False)
                
        elif label=='rate_coeff':
            if k_inp==[]:
                k_inp=self.k  #Uses the default rate constants alreayd specified from Param File
                datk = k_inp
            else:
                datk = k_inp #Uses user inputed rate constants
            datk_df = pd.DataFrame(datk)
            klength_entry = len(datk_df.iloc[:])
            if klength_entry != len(self.k) :  #Check to see length of rate coefficients array is as expected
                raise Exception('Number of rate coefficients should be:',len(self.k))
            else:
                if Name==None:
                    datk_df.to_csv(label+'.csv', encoding='utf-8', index=False)
                else:
                    datk_df.to_csv(Name, encoding='utf-8', index=False)
    #------------------------------------------------------------------------------------------------------------------------------
    #Function responsible for plotting
    #------------------------------------------------------------------------------------------------------------------------------    
    def plotting(self,sol,solt,label):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        
        for i in np.arange(len(sol[0,:])):
            ax.plot(solt, sol[:,i])
            
        if label=='rates_p':     
            ax.legend(self.Atomic.columns.values[1:],fontsize=10, loc='upper right',facecolor='white', edgecolor ='black', framealpha=1)
            ax.set_xlabel('Time, t, [s]')
            ax.set_ylabel(r"Rates of Production, $R_i$")
            ax.set_title('Rates of production versus Time')
            
        elif label=='rates_r':
            ax.legend(np.array(self.Stoich.iloc[:,0]),fontsize=10, loc='upper right',facecolor='white', edgecolor ='black', framealpha=1)
            ax.set_xlabel('Time, t, [s]')
            ax.set_ylabel(r"Rates of Reaction, $r_i$")
            ax.set_title('Rates of reaction versus Time')
            
        elif label=='coverages':
            ax.legend(self.Atomic.columns.values[1+len(self.P):],fontsize=10, loc='upper right',facecolor='white', edgecolor ='black', framealpha=1)
            ax.set_xlabel('Time, t, [s]')
            ax.set_ylabel(r"Coverage, $\theta_i, [ML]$")
            ax.set_title('Coverages versus Time')
#------------------------------------------------------------------------------------------------------------------------------    
class Fitting:    
    def __init__(self,Input_csv,Atomic_csv,Stoich_csv,Param_Guess_csv,Input_Type='iCovg_iRates'): #Inputs necessary to initialize the MK Model
        self.MKM = MKModel(Atomic_csv,Stoich_csv,Param_Guess_csv) #Initializing the MF-MK Model
        
        self.Input = pd.read_csv(Input_csv)
        self.Atomic = pd.read_csv(Atomic_csv)     #Opening/Reading the Atomic input file needed to be read
        self.Stoich = pd.read_csv(Stoich_csv)    #Opening/Reading the Stoichiometric input file needed to be read
        self.Param_Guess = pd.read_csv(Param_Guess_csv)     #Opening/Reading the Parameter of guess input file needed to be read       
        self.k = self.kextract()    #Extracting the rate constants from the Param File (Note that format of the Param File is crucial) #To be used as initial guess
        self.P ,self.Temp = self.set_rxnconditions() #Setting reaction conditions (defaulted to values from the Param File but can also be set mannually )
        self.Ti,self.Tf=self.set_limits_of_integration() #Sets the range of time needed to solve for the relavant MK ODEs, defaults to 0-6e6 but can also be manually set
        self.rate_const_correction = 'None' #Accounting for correction to the rate constants (i.e. enhancing the mean field approximation)
        self.MKM.rate_const_correction = self.rate_const_correction
        self.Input_Type = Input_Type
        self.PARAM = self.set_params
        self.BG_matrix='auto' #Bragg williams constant matrix
        self.Coeff = self.Coeff_extract() #Extracting the coverage dependance coefficients
        self.init_cov=self.set_initial_coverages() #Sets the initial coverage of the surface species, defaults to zero coverage but can also be set manually
        self.n_extract = 0.5 #Nummber of points to be extracted from the input file #Defaulted to 0.5 of the total points
        self.status='Waiting' #Used to observe the status of the ODE Convergence
        self.label='None'   #Used to pass in a label so as to know what kind of figure to plot
        #Output: self.fitted_k  #Can be used to extracted an array of final fitted rate parameters
        if self.Input_Type not in ['iCovg','iCovg_iRates']:
            raise Exception('Input type specified is not recognised.\n Please make sure your input type is among that which is acceptable')
    #------------------------------------------------------------------------------------------------------------------------------   
    def check_massbalance(self,Atomic,Stoich): #Function to check if mass is balanced
        return self.MKM.check_massbalance(self.Atomic,self.Stoich)
    #------------------------------------------------------------------------------------------------------------------------------    
    def check_coverages(self,vec):  #Function to check if the coverages being inputted make sense (Note in this code empty sites are not inputted, they're calculated automatically)
        return self.MKM.check_coverages(vec)
    #------------------------------------------------------------------------------------------------------------------------------    
    def Pextract(self): #Function used for extracting pressures from the Param File
        return self.MKM.Pextract()
    #------------------------------------------------------------------------------------------------------------------------------
    def kextract(self): #Function used for extracting rate constants from the Param File
        return self.MKM.kextract()
    #------------------------------------------------------------------------------------------------------------------------------
    def set_initial_coverages(self,init=[]): #empty sites included at the end of the code
        return self.MKM.set_initial_coverages()
    #------------------------------------------------------------------------------------------------------------------------------    
    def set_rxnconditions(self,Pr=None,Temp=None): #Function used for setting the reaction Pressure and Temperature (Currently Temperature is not in use)
        self.P,self.Temp = self.MKM.set_rxnconditions(Pr,Temp)
        return self.P,self.Temp
    #------------------------------------------------------------------------------------------------------------------------------
    def set_limits_of_integration(self,Ti=0,Tf=6e6): #Function used for setting the time limits of integration 
        self.Ti,self.Tf=self.MKM.set_limits_of_integration(Ti,Tf)
        return self.Ti,self.Tf
    #------------------------------------------------------------------------------------------------------------------------------
    def set_params(self,k=[]):
        if k==[]:
            k = self.k

        params = Parameters()
        for i in np.arange(len(k)):
            params.add('k'+str(i+1),value=k[i],min=0)

        return params
    #------------------------------------------------------------------------------------------------------------------------------
    def Coeff_extract(self):
        return self.MKM.Coeff_extract()
    #------------------------------------------------------------------------------------------------------------------------------    
    def extract(self,inp_array=[],InputType=[]): #Note: Input and output both include time vector
        Ncs = len(self.Stoich.iloc[0,:])-len(self.Pextract())-1 #No. of Surface species
        Ngs = len(self.Pextract()) #No. Gaseous Species

        #Setting up the type of input being extracted
        if InputType==[]:
            InputType=self.Input_Type
        else:
            self.Input_Type=InputType

        #Extracting Relevant Features
        if InputType=='iCovg': #Instantaneous Coverages and Rates
            if inp_array==[]: #If no input array is inserted, then the default input is used
                Input_time_array = self.Input.iloc[:,0].to_numpy() #Time
                Input_covg_array = self.Input.iloc[:,1:Ncs+1].to_numpy() #Coverage
                lnt = len(Input_time_array) #length of the (default) input array
            else:
                Input_time_array = inp_array[:,0] #Time
                Input_covg_array = inp_array[:,1:Ncs+1] #Note that the inputed array must include time    
                lnt = len(Input_time_array) #length of the inputed array

        elif InputType=='iCovg_iRates': #Instantaneous Coverages and Rates
            if inp_array==[]: #If no input array is inserted, then the default input is used
                Input_time_array = self.Input.iloc[:,0].to_numpy() #Time
                Input_covg_array = self.Input.iloc[:,1:Ncs+1].to_numpy() #Coverage
                Input_rates_array = self.Input.iloc[:,-Ngs:].to_numpy() #Rates
                lnt = len(Input_time_array)
            else:
                Input_time_array = inp_array[:,0] #Time
                Input_covg_array = inp_array[:,1:Ncs+1] #Note that the inputed array must include time    
                Input_rates_array = inp_array[:,-Ngs:] #Note that the inputed array must include time    
                lnt = len(Input_time_array) #length of the inputed array

        if self.n_extract<=1 and self.n_extract>0:
            n_extr = int(self.n_extract*np.shape(Input_covg_array)[0]) #Calculating number of points to be extracted based on the percentage entered (eg. 0.7 = 70%)
            if n_extr<=1: #Checking to see if calculating if number of points to be excited is less than or equal to 1
                raise Exception('Percentage of input values selected is too low.')
                
            print(self.n_extract*100,'% of the Input dataset is being extracted for fitting (i.e',n_extr,'points are being extracted for fitting)\n')
        elif self.n_extract>1:
            n_extr = int(self.n_extract)
            print(n_extr,' points in the Input dataset are being extracted for fitting\n')
        else:
            raise Exception('Please enter a value from 0 to 1 to indicate percent of input data or greater than 1 for a specific positive number to indicate the desired number of points to be extracted.')
               
        dist = len(Input_time_array[::round(lnt/n_extr)]) #length to be used to intilaize empty array   

        Time_Inp = np.empty((dist,1))
        Covg_Inp = np.empty((dist,Ncs)) #Extracted n values from input
        Rates_Inp = np.empty((dist,Ngs))

        if InputType=='iCovg' or InputType=='iCovg_iRates':  
            Time_Inp = Input_time_array[::round(lnt/n_extr)]
            for i in np.arange(Ncs): #looping over Number of species 
                if np.isnan(Input_covg_array[:,i]).any() == True:
                    raise Exception('Check Number of Surface_species Ncs is correct; Check to see correct method has been chosen; Check to see if Input format is correct')
                Covg_Inp[:,i]=Input_covg_array[:,i][::round(lnt/n_extr)]
        
        if InputType=='iCovg_iRates':
            for i in np.arange(Ngs): 
                Rates_Inp[:,i] = Input_rates_array[:,i][::round(lnt/n_extr)]

        if InputType=='iCovg':
            return Time_Inp,Covg_Inp
        elif InputType=='iCovg_iRates':
            return Time_Inp,Covg_Inp,Rates_Inp   
    #------------------------------------------------------------------------------------------------------------------------------    
    def paramorderinfo(self,Source='Model',Param='Pressure'):  
        enablePrint()
        Ngs = len(self.Pextract())
        Ncs = len(self.Stoich.iloc[0,:])-Ngs-1 #No. of Surface species
        if Source=='Model':
            if Param=='Pressure':
                print('\n Order for Input Pressures [Pa]:')
                Pr_header = np.array(self.Stoich.columns[1:Ngs+1])
                x = []
                for i in np.arange(len(Pr_header)): x.append(i)
                list = [x]
                print(pd.DataFrame(list,columns=Pr_header, index=['Array order']))
            elif Param=='Coverage':
                print('\n Order for Input Coverages (Transient and Steady State) [ML]:')
                Covg_header = np.array(self.Stoich.columns[Ngs+1:])
                x = []
                for i in np.arange(len(Covg_header)): x.append(i)
                list = [x]
                print(pd.DataFrame(list,columns=Covg_header, index=['Array order']))
            elif Param=='Rates_Production':
                print('\n Order for Input Rates of Production (Transient and Steady State) [TOF]:')
                Rp_header = np.array(self.Stoich.columns[1:])
                x = []
                for i in np.arange(len(Rp_header)): x.append(i)
                list = [x]
                print(pd.DataFrame(list,columns=Rp_header, index=['Array order']))
            elif Param=='Rates_Reaction':
                print('\n Order for Input Rates of Reactions (Transient and Steady State) [TOF]:')
                Rr_header = np.array(self.Stoich.iloc[:,0])
                x = []
                for i in np.arange(len(Rr_header)): x.append(i)
                list = [x]
                print(pd.DataFrame(list,columns=Rr_header, index=['Array order']))
            elif Param=='Rate_Constants':
                print('\n Order for Input Rate Constants [1/s]:')
                params_header = []
                for i in np.arange(len(self.k)):
                    params_header.append('k'+str(i+1))
                x=[]
                for i in np.arange(len(params_header)): x.append(i)
                list = [x]
                print(pd.DataFrame(list,columns=params_header, index=['Array order']))
        blockPrint()
        return 
    #------------------------------------------------------------------------------------------------------------------------------    
    def normalize(self,Ext_inp=[]):  #Note: Input and output both include time vector
        if Ext_inp==[]:
            inp=self.extract(inp_array=[])
            
        else:
            inp=self.extract(inp_array=Ext_inp)
        
        Norm_inp = np.empty(np.shape(inp))
        for i in np.arange(len(inp[0,:])):
            if all(j < 1e-12 for j in inp[:,i]):
                print('An essentially zero vector is present and therefore cant be normalized. The same vector has been returned.\n')
                Norm_inp[:,i] = inp[:,i]
            else:
                mi = min(inp[:,i])
                ma = max(inp[:,i])
                Norm_inp[:,i]=(inp[:,i]-mi)/(ma-mi)
        print('Input dataset has been normalized for fitting')
        return Norm_inp
    #------------------------------------------------------------------------------------------------------------------------------    
    def denormalize(self,Ext_inp_denorm=[]):
        if Ext_inp_denorm==[]:
            norm_inp = self.normalize()  #Using the default normalized data from input csv file
            
        else:
            norm_inp = self.normalize(Ext_inp=Ext_inp_denorm) #Using the concatenated sol and solt arrays from analysis (i.e soldat = np.insert(sol,0,solt,axis=1)   #Merging time and parameters)
            
        inp = self.extract(inp_array=[]) #Exctracted input array from Input csv file
        
        Denorm_inp = np.empty(np.shape(inp))
        for i in np.arange(len(inp[0,:])):
            mi = min(inp[:,i])
            ma = max(inp[:,i])
            Denorm_inp[:,i]=(norm_inp[:,i]*(ma-mi)) + mi
        return Denorm_inp
    #------------------------------------------------------------------------------------------------------------------------------    
    # Cost/Minimization Functions
    #------------------------------------------------------------------------------------------------------------------------------    
    # Rate_functions that generates combination of constrained coverages _ to be mininimized to obtain rate parameters using a curvefit method
    #------------------------------------------------------------------------------------------------------------------------------    
    def kinetic_output_iCovg_iRates(self,x,*fit_params): #covg and steady state rates of prod
        fit_params_array = np.array(fit_params)
        Ncs = len(self.Stoich.iloc[0,:])-len(self.Pextract()) #No. of Surface species
        Ngs = len(self.MKM.Pextract()) #No. of gaseous species
        rw = len(self.k)

        self.MKM.k = fit_params_array       

        input_time,input_covg,input_rate = self.extract()
        inp_init_covg = input_covg[0,:]  #Used to make sure the intial coverages match the input

        covg_sol,covg_t= self.MKM.solve_coverage(t=[0,input_time[-1]],initial_cov=inp_init_covg,Tf_eval=input_time,full_output=False) #Uses MKM.getODEs, but the inclass solve_coverage to add custom time dependancies
        ratep_sol, ratep_t = self.MKM.solve_rate_production(tf=None,Tf_eval=input_time) #instantaneous rates of production of all species
        gratep_sol = ratep_sol[:,:Ngs] #Extracting instantaneous gaseous species rates of production  #Note: gas species are the ordered first 

        kin_output_covg = np.reshape(covg_sol,covg_sol.size)
        kin_output_gratep = np.reshape(gratep_sol,gratep_sol.size)
        kin_output = np.concatenate((kin_output_covg,kin_output_gratep))

        return kin_output
    #------------------------------------------------------------------------------------------------------------------------------    
    def rate_func_iCovg_iRates(self,Params,x,y): #covg and steady state rates of prod
        Ncs = len(self.Stoich.iloc[0,:])-len(self.Pextract())-1 #No. of Surface species
        Ngs = len(self.MKM.Pextract()) #No. of gaseous species
        colmn = len(self.Stoich.iloc[0,1:]) - len(self.P) - 1 #Number of columns (i.e rate coefficients = no. of surface species being investigated)
        rw = len(self.k)

        #x = input times   ; y = input coverages (to be fitted)
        v = Params.valuesdict()
        fit_params_array = np.empty(len(self.k))
        for i in np.arange(len(self.k)):
            fit_params_array[i] = v['k'+str(i+1)]

        self.MKM.k = fit_params_array       

        input_time,input_covg,input_rate = self.extract()
        inp_init_covg = input_covg[0,:]  #Used to make sure the intial coverages match the input

        covg_sol,covg_t= self.MKM.solve_coverage(t=[0,input_time[-1]],initial_cov=inp_init_covg,Tf_eval=input_time,full_output=False) #Uses MKM.getODEs, but the inclass solve_coverage to add custom time dependancies
        ratep_sol,ratep_t = self.MKM.solve_rate_production(tf=None,Tf_eval=input_time) #steady state rates of production of all species
        gratep_sol = ratep_sol[:,:Ngs] #Extracting instantaneous gaseous species rates of production  #Note: gas species are the ordered first

        kin_output_covg = np.reshape(covg_sol,covg_sol.size)
        kin_output_gratep = np.reshape(gratep_sol,gratep_sol.size)
        kin_output = np.concatenate((kin_output_covg,kin_output_gratep))

        return kin_output - y 
    #------------------------------------------------------------------------------------------------------------------------------    
    def error_func_0(self,fit_params):
        fit_params_array = np.array(fit_params)
        colmn = len(self.Stoich.iloc[0,1:]) - len(self.P) - 1 #Number of columns/species (i.e rate coefficients = no. of surface species being investigated) #Exckuding empty sites
        og = self.extract() #Original input #NO NORMALIZING
        klen = len(self.k)

        self.MKM.k = fit_params_array   
        
        input_time=self.extract()[:,0]
        inp_init_covg = self.extract()[0,1:-1]
        sol,solt= self.solve_coverage(t=[0,input_time[-1]],initial_cov=inp_init_covg,Tf_eval=input_time,full_output=False) #Uses MKM.getODEs, but the inclass solve_coverage to add custom time dependancies
        soldat = np.insert(sol,0,solt,axis=1)   #Merging time and parameters
        #Norm_sol = self.normalize(Ext_inp=soldat)
        
        rw = len(soldat[:,0]) #Coverages
        colmn = colmn+1 #Including empty sites
        
        w = np.ones(colmn) #weight of RSS Residual Sum of Squares #constant across species #specified by user- using self.min_weight
        error_matrix = np.zeros((rw,colmn))

        #Gnerating sum weight by an inverse of max covg function
       # max_covgs = np.zeros(colmn) #empty set to hold maximum coverages used to auto-select weight
        #for j in np.arange(colmn):
         #   max_covgs[j] = max(og[:,j+1]) #j+1 to skip the time
          #  if max_covgs[j] < 0.0001: #if statement to prevent over-emphasizing a species
           #     max_covgs[j] = 0.0001
            #w[j] = 1/(4*(max_covgs[j]))      

        for i in np.arange(rw):
            for j in np.arange(colmn):
                error_matrix[i,j]=(og[i,(j+1)] - soldat[i,j+1])**2
        
        colmn_sumn = error_matrix.sum(axis=0)
        error = 0
        for j in np.arange(colmn):
            error = error + w[j]*colmn_sumn[j]

        return error
    #------------------------------------------------------------------------------------------------------------------------------    
    def error_func_1(self,fit_params):
        fit_params_array = np.array(fit_params)
        colmn = len(self.Stoich.iloc[0,1:]) - len(self.P) - 1 #Number of columns (i.e rate coefficients = no. of surface species being investigated)
        rw = len(self.k)
        og = self.extract() #Original input
        
        self.MKM.k = fit_params_array
            
        input_time=self.extract()[:,0]
        inp_init_covg = self.extract()[0,1:-1]
        sol,solt= self.solve_coverage(t=[0,input_time[-1]],initial_cov=inp_init_covg,Tf_eval=input_time) #Uses MKM.getODEs, but the inclass solve_coverage to add custom time dependancies
        soldat = np.insert(sol,0,solt,axis=1)   #Merging time and parameters
        #Norm_sol = self.normalize(Ext_inp=soldat)
        
        return np.sum((og-soldat)**2)
    #------------------------------------------------------------------------------------------------------------------------------    
    def CI95(self,fvec, jac): #Function to find confidence interval ############################----NEEDS FIXING----####################################################
        #Returns the 95% confidence interval on parameters
        
        rss = np.sum(fvec**2) # residual sum of squares
        
        n, p = jac.shape     # number of data points and parameters
       
        nmp = n - p          # the statistical degrees of freedom
        
        ssq = rss / nmp      # mean residual error
        
        J = np.matrix(jac)   # the Jacobian
        
        c = np.linalg.inv(J*np.transpose(J))       # covariance matrix
        
        print(J*np.transpose(J))
        print(c)
        pcov = c * ssq       # variance-covariance matrix.
        # Diagonal terms provide error estimate based on uncorrelated parameters.
        
        err = np.sqrt(np.diag(np.abs(pcov))) * 1.96  # std. dev. x 1.96 -> 95% conf
        # Here err is the full 95% area under the normal distribution curve. 
        return err
    #------------------------------------------------------------------------------------------------------------------------------    
    # Optimizers/Fitting Functions
    #------------------------------------------------------------------------------------------------------------------------------     
    def minimize_fun(self,method,max_nfev = None):

        if self.Input_Type=='iCovg_iRates':
            cost_function = self.rate_func_iCovg_iRates
            time_values,covg_values,ratep_values = self.extract() # Input Data
        
            x_values = time_values # Input Time variables (Independent Variable) (eg. KMC Time)

            y_values_covg = np.reshape(covg_values,covg_values.size) # Input Dependent variable(s) (eg. KMC coverages)

            y_values_gratep = np.reshape(ratep_values,ratep_values.size) # Input Dependent variable(s) (eg. KMC rates_p)

            y_values = np.concatenate((y_values_covg,y_values_gratep)) #Including the instantaneous rates of productions to be compared with/error minimized
            
        # parameters = self.PARAM
        parameters = Parameters()
        for i in np.arange(len(self.k)):
            parameters.add('k'+str(i+1),value=self.k[i],min=0)

        fitted_params = minimize(self.rate_func_iCovg_iRates, parameters, args=(x_values,y_values), method=method, max_nfev = max_nfev)

        vec_fitted_param = np.empty(len(self.k))
        for i in np.arange(len(self.k)):
            vec_fitted_param[i]= fitted_params.params['k'+str(i+1)].value
        
        vec_fitted_param_covariance = fitted_params.covar
        return vec_fitted_param, vec_fitted_param_covariance
    #------------------------------------------------------------------------------------------------------------------------------ 
    def minimizer_fit_func(self,method,gtol,ftol,maxfun,maxiter,tol,xatol,fatol,adaptive):
        values = self.extract()

        x_values = values[:,0]
        y_values = np.reshape(values[:,1:],values[:,1:].size)
        
        #Setting Bounds
        #max K Guess parameters
#---------# Bound generation method 1: Scalar Multiples#---------#---------#---------#---------#---------#---------             
        sc = 1e3 #scaling value
        c = len(self.Stoich.iloc[0,1:]) - len(self.P) - 1 #Number of columns (i.e no. of surface species being investigated)
        
        # if self.CovgDep==True:
        #     initial_vals = np.concatenate((self.k,self.Coeff.flatten(order='F')))
        #     mkval = initial_vals*sc #max coeffvals
        #     n = len(initial_vals)
        #     lnk = len(self.k)
        #     bounds1 = np.empty([c*n,2])
        #     for i in range(n):
        #         bounds1[i] = (0,mkval[i])  #Rate constants
        #         bounds1[lnk+i] = (-mkval[lnk+i],mkval[lnk+i]) #Rate coefficients
        
        # elif self.CovgDep==False:
        initial_vals = self.k
        mkval = initial_vals*sc #max coeffvals
        n = len(initial_vals)
        bounds1 = np.empty([n,2])
        for i in range(n):
            bounds1[i] = (0,mkval[i]) #Rate constants
#---------#---------#---------#---------#---------#---------#---------#---------#---------#---------#---------#---------    
#---------# Bound generation method 2: Maintaining the order of magnitude of the guess values#---------#------#---------#---------                         
        # if self.CovgDep==True:
        #     initial_vals = np.concatenate((self.k,self.Coeff.flatten(order='F')))
        #     n = len(initial_vals)
        #     bounds2 = np.empty([c*n,2])
        #     for i in range(n):
        #         expon = math.floor(math.log(initial_vals[i], 10)) #order of magnitude of the guess values
        #         upper = 1*10**(expon+1)
        #         lower = 9*10**(expon-1)
        #         bounds2[i] = (lower,upper)  #Rate constants
        #         bounds2[lnk+i] = (lower,upper) #Rate coefficients
        
        # elif self.CovgDep==False:
        #     initial_vals = self.k
        #     n = len(initial_vals)
        #     bounds2 = np.empty([n,2])
        #     for i in range(n):
        #         expon = math.floor(math.log(initial_vals[i], 10)) #order of magnitude of the guess values
        #         upper = 1*10**(expon+1)
        #         lower = 9*10**(expon-1)
        #         bounds2[i] = (lower,upper)  #Rate constants
 #---------#---------#---------#---------#---------#---------#---------#---------#---------#---------#---------#---------#---------         
        
        from numdifftools import Jacobian#, Hessian
        from autograd import jacobian #, hessian
        import autograd.numpy as anp
        import jax.numpy as jnp
        from jax import jacfwd
        
        #USING NUMDIFFTOOLS#---------#---------#---------#
        def jacf(x):
            jac = Jacobian((lambda x: self.error_func_0(x)),method='forward',order=3)(x).ravel()
            # enablePrint()
            # print(jac)
            # blockPrint()
            return jac
        
        #****** NOT WORKING:***********************************
        #USING AUTOGRAD #---------#---------#---------#
        def jacf1(x):
            x = anp.array(x)
            jac = jacobian(lambda x: self.error_func_0(x))(x).ravel()
            # enablePrint()
            # print(jac)
            # blockPrint()
            return jac
        
        
        #USING JAX #---------#---------#---------#
        def jacf2(x):
            x = jnp.array(x)
            jac = jacfwd(lambda x: self.error_func_0(x))(x).ravel()
            # enablePrint()
            # print(jac)
            # blockPrint()
            return jac
        #******#******#******#******#******#******#******#******#******   
        if method=='nelder-mead':
            jacfunc = False
        else:
            jacfunc = jacf
        result = optimize.minimize(self.error_func_0, initial_vals
                                                    ,method=method, bounds=bounds1,tol=tol
                                                    ,jac= jacfunc
                                                    ,options={'xatol': xatol, 'fatol': fatol,'gtol': gtol 
                                                              ,'ftol': ftol,'maxfun': maxfun,'disp': False
                                                              ,'maxiter': maxiter, 'adaptive':adaptive})
      
        # result = optimize.differential_evolution(self.error_func,bounds=bounds1,tol=1e-3,seed=45
        #                                             ,maxiter=2,disp=False, polish=True,workers=1)
        
        return result
    #------------------------------------------------------------------------------------------------------------------------------
    def ML_data_gen_0(self,n): ##degree of change is uniform across rates
        a = len(self.Stoich.iloc[0,1:]) - len(self.P) - 1 #Number of columns (i.e rate coefficients = no. of surface species being investigated)
        b = len(self.k)
        og = self.extract() #Original input
        
        # if self.CovgDep==True:
        #     rate_cvals = np.concatenate((self.k,self.Coeff.flatten(order='F')))
        # elif self.CovgDep==False:
        rate_cvals = self.k
            
        n = 20
        
        # Getting Rate coefficient matrix
        Rate_Coeff = np.zeros((n,b))
        con = 100   #Starting at 200% increased, loop would keep going down depending on size of n
        perc = con
        for i in np.arange(n):
            Rate_Coeff[i,:] = (rate_cvals)*(1+con/100)
            con = con - (perc/n)*(1+(2*perc/2e2))
            if any(a < 0 for a in Rate_Coeff[i,:]):
                raise Exception('Error, rate constants generated are negative')

        # Getting coverage matrix (rank 3 tensor)
        Covg = np.zeros((n,np.shape(og[:,1:])[0],np.shape(og[:,1:])[1]))
        input_time=og[:,0]
        inp_init_covg = og[0,1:-1]
        for i in np.arange(n):
            
            # if self.CovgDep==False:
            self.MKM.k = Rate_Coeff[i,:]
            # elif self.CovgDep==True:
            #     self.MKM.k = Rate_Coeff[i,:b]
            #     self.MKM.Coeff = np.reshape(Rate_Coeff[i,b:],(b,a))
                
            sol,solt= self.solve_coverage(t=[0,input_time[-1]],initial_cov=inp_init_covg,Tf_eval=input_time)
            Covg[i,:,:] = sol
            # print(Rate_Coeff)
            # print(Covg)
        return Rate_Coeff,Covg
    #------------------------------------------------------------------------------------------------------------------------------
    def ML_data_gen_1(self,n): #Magnitude remains the same across the entire rate constant matrix
        a = len(self.Stoich.iloc[0,1:]) - len(self.P) - 1 #Number of columns (i.e rate coefficients = no. of surface species being investigated)
        klen = len(self.k)
        og = self.extract() #Original input

        # if self.CovgDep==True:
        #     rate_cvals = np.concatenate((self.k,self.Coeff.flatten(order='F')))
        # elif self.CovgDep==False:
        rate_cvals = self.k
            
        # enablePrint() #Enabling printing   
        # Getting Rate coefficient matrix
        Rate_Coeff = np.zeros((n,klen))
        digits = list(range(1,10)) #range of numbers from 1 to 9
        
        #Generating permutations of numbers from 1 to 9 to use in generating values to use for rate_coeffs 
        from itertools import permutations
        from itertools import product
        from itertools import islice 
        
        #Note: n = number of rows for rate_coeff matrix, klen = number of rate constants
        values = np.array(list(islice(permutations(digits),1,n*klen))).flatten() #Permutations don't allow for repitition
        # values = np.array(list(islice(product(digits,repeat=len(digits)),1,n*klen))).flatten()  #Allowing repitition
        count = 0
        
        for i in np.arange(n):  #looping through the desired number of rows of dataset 
            for j in np.arange(klen):
                expon = math.floor(math.log(self.k[j], 10)) #order of magnitude of the guess values
                Rate_Coeff[i,j]= values[count]*10**(expon)
                count = count + 1
        
            
        # Getting coverage matrix (rank 3 tensor)
        Covg = np.zeros((n,og.shape[0],np.shape(og[:,1:])[1]))

        input_time=og[:,0]
        inp_init_covg = og[0,1:-1]
        for i in np.arange(n):
            
            # if self.CovgDep==False:
            self.MKM.k = Rate_Coeff[i,:]
            # elif self.CovgDep==True:
            #     self.MKM.k = Rate_Coeff[i,:klen]
            #     self.MKM.Coeff = np.reshape(Rate_Coeff[i,klen:],(klen,a))
                
            sol,solt= self.solve_coverage(t=[0,input_time[-1]],initial_cov=inp_init_covg,Tf_eval=input_time)
            Covg[i,:,:] = sol
        
        return Rate_Coeff,Covg
    #------------------------------------------------------------------------------------------------------------------------------
    def ML_data_gen_2(self,n): #For Pressure variation *********************
        a = len(self.Stoich.iloc[0,1:]) - len(self.P) - 1 #Number of columns (i.e rate coefficients = no. of surface species being investigated)
        Plen = len(self.P)
        og = self.extract() #Original input
        klen = len(self.k)
        # if self.CovgDep==True:
        #     rate_cvals = np.concatenate((self.k,self.Coeff.flatten(order='F')))
        # elif self.CovgDep==False:
        rate_cvals = self.k
             
        # Getting Rate coefficient matrix
        Pressures = np.zeros((n,Plen))
        digits = list(range(1,10)) #range of numbers from 1 to 9
        
        #Generating permutations of numbers from 1 to 9 to use in generating values to use for rate_coeffs 
        from itertools import permutations
        from itertools import product
        from itertools import islice 
        
        #Note: n = number of rows for rate_coeff matrix, klen = number of rate constants
        # values = np.array(list(islice(permutations(digits),1,n*klen))).flatten() #Permutations don't allow for repitition
        values = np.array(list(islice(product(digits,repeat=len(digits)),1,n*klen))).flatten()  #Allowing repitition
        count = 0
        
        for i in np.arange(n): #looping through the desired number of rows of dataset 
            for j in np.arange(Plen):
                expon = math.floor(math.log(self.P[j], 10)) #order of magnitude of the guess values
                Pressures[i,j]= values[count]*10**(expon)
                count = count + 1
        
            
        # Getting coverage matrix (rank 3 tensor)
        Covg = np.zeros((n,og.shape[0],np.shape(og[:,1:])[1]))

        input_time=og[:,0]
        inp_init_covg = og[0,1:-1]
        for i in np.arange(n):
            
            self.P = Pressures[i]
                
            sol,solt= self.solve_coverage(t=[0,input_time[-1]],initial_cov=inp_init_covg,Tf_eval=input_time)
            Covg[i,:,:] = sol

        return Rate_Coeff,Covg
    #------------------------------------------------------------------------------------------------------------------------------
    def ML_model_predict(self,Covg_fit,n,filename,mdl):
        ML_Dataset = self.ML_data_gen_1(n=n)

        #y = Rate_Coeff => trying to predict
        #X = Covg => the dataset
        a = len(self.Stoich.iloc[0,1:]) - len(self.P)  #Number of surface species being investigated
        if filename!=None and filename[-5:]!='.xlsx':  #Making sure that the Name entered has .csv attachment
            raise Exception("Name entered must end with .xlsx ; Example ML_dataset_1.xlsx")
            
        if os.path.exists(filename) != True or n!=len(pd.read_excel('./'+filename, sheet_name='Rate_Coeffs')) or len(self.extract()) != len(pd.read_excel('./'+filename, sheet_name='Coverages')): #if the ML_data excel file doesnt exists or it doesn't have same no. of rows as the number specified, it replaces the dataset with the desired one
            #Making the ML_data excel file and saving it into the directory
            Rate_Coeff,Covg = ML_Dataset
            Rate_Coeff_df = pd.DataFrame(Rate_Coeff)
            Covg_df = pd.DataFrame(Covg.reshape(Covg.shape[0],math.prod(Covg.shape[1:])))
            writer = pd.ExcelWriter('./'+filename, engine='xlsxwriter')
            Sheets = {'Rate_Coeffs': Rate_Coeff_df, 'Coverages': Covg_df}
            for sheet_name in Sheets.keys():
                Sheets[sheet_name].to_excel(writer, sheet_name=sheet_name,index=False)
            writer.save()
            #Using the values from function directly to save time.
            y,X = Rate_Coeff,Covg
            
        elif os.path.exists(filename) == True and n==len(pd.read_excel('./'+filename, sheet_name='Rate_Coeffs')) : #If the ML_data excel file already exists, ML dataset wont be regenerated
            Rate_Coeff_df = pd.read_excel('./'+filename, sheet_name='Rate_Coeffs')
            Covg_df = pd.read_excel('./'+filename, sheet_name='Coverages')
            Rate_Coeff = Rate_Coeff_df.to_numpy()
            Covg = Covg_df.to_numpy().reshape(Rate_Coeff.shape[0],-1,a)
            y,X = Rate_Coeff,Covg
     
        X=X.reshape(X.shape[0],-1)
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import KFold
        from sklearn.model_selection import cross_val_score

        #mdl = model selection
        
        if mdl == 'MLPRegressor':
            enablePrint() #Enabling printing
            print('-Using Algorithm: MLPRegressor | (FeedForward) Neural Network:\n')
            No_H_nodes_per_layer = 10
            print('Number of Hidden layer nodes per layer : ',No_H_nodes_per_layer)
            No_H_layers = 3
            print('Number of Hidden layers: ',No_H_layers)
            blockPrint() #Disabling printing
            
            hidden_layers = No_H_nodes_per_layer*np.ones(No_H_layers) 
            hidden_layer_sizes = tuple(tuple(int(item) for item in hidden_layers))
            regr = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                activation='tanh', 
                                solver='adam',random_state=42, 
                                max_iter=5000,alpha=5e-6,
                                learning_rate='adaptive',tol=5e-7,shuffle=True)
            

        elif mdl == 'KNeighborsRegressor':
            enablePrint() #Enabling printing
            print('-Using Algorithm: K Nearest Neighbor Regressor:\n')
            blockPrint() #Disabling printing
            
            regr= KNeighborsRegressor(n_neighbors=5,weights='uniform',
                                      algorithm='auto', leaf_size=40, 
                                      p=2, metric='cosine')
            
        elif mdl == 'DecisionTreeRegressor':
            enablePrint() #Enabling printing
            print('-Using Algorithm: Decision Tree Regressor:\n')
            blockPrint() #Disabling printing
            
            regr = DecisionTreeRegressor(criterion='mse',splitter='best',
                                         min_samples_leaf = 1,max_features='auto',
                                         min_samples_split=10,random_state=42)
        
        elif mdl == 'RandomForestRegressor':
            enablePrint() #Enabling printing
            print('-Using Algorithm: Random Forest Regressor:\n')
            blockPrint() #Disabling printing
            
            regr = RandomForestRegressor(n_estimators=300,min_samples_leaf = 1,
                                         max_features='auto',min_samples_split=10,
                                         random_state=42,criterion='poisson')  
         
        # Defining the model
        model = regr
        
        # Fitting the model
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, random_state=40)
        model.fit(X_train, Y_train)
        
        # Evaluating the model
        from sklearn.metrics import mean_squared_error
        Y_pred = model.predict(X_test)
        
        
        #K fold cross validation
        # accuracy_model = []

        # from sklearn.metrics import accuracy_score
        # kf = KFold(n_splits=5)
        # for train_index, test_index in kf.split(X):
        #     # Split train-test
        #     X_train, X_test = X[train_index], X[test_index]
        #     y_train, y_test = y[train_index], y[test_index]
        #     # Train the model
        #     model = regr.fit(X_train, y_train)
        #     # Append to accuracy_model the accuracy of the model
        #     accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True)*100)
            
    
        enablePrint() #Enabling printing
        # print('Scores:')
        # scores = pd.DataFrame(accuracy_model,columns=['Scores'])
        # import seaborn as sns
        # sns.set(style="white", rc={"lines.linewidth": 3})
        # sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="Scores",data=scores)
        # plt.show()
        # sns.set()
        MSE = mean_squared_error(Y_test, Y_pred, multioutput='raw_values')
        print("The Model Mean Squared Errors: \n {}".format(MSE))
        # blockPrint() #Disabling printing
   
        # Making the prediction
        actual_pred_values = model.predict(Covg_fit.reshape(1, -1))
        fit_params = abs(actual_pred_values.flatten())
        
        return fit_params    
    #------------------------------------------------------------------------------------------------------------------------------
    def fitting_rate_param(self,option='min',plot=False,plot_norm=False,method_min='least_squares',method_cmin='least_squares',method_fit='leastsq',weights = None,mdl='MLPRegressor'
                           ,maxfev=1e5,xatol=1e-4,fatol=1e-4,adaptive=False,tol = 1e-8,xtol=1e-8,ftol=1e-8,gtol=1e-8,maxfun=1e6,maxiter=1e5,weight=1e0,n=40,filename='ML_dataset.xlsx'):
        #n is the number of rows worth of ML data, if it is changed and the present data has different rows, a new dataset will be generated with n rows 
        
        colmn = len(self.Stoich.iloc[0,1:]) - len(self.P) - 1 #Number of columns (i.e no. of surface species being investigated)
        index = list(string.ascii_lowercase)[:colmn]
        if self.Input_Type=='iCovg_iRates':
            og_time,og_covg,og_srate_p = self.extract()
        # blockPrint() #Preventing reprinting in jupyter
        if option=='min':
        
            print('Performing fitting using LMFIT package:')
            print("-"*50)
            print('-Using Method:',method_min)
            blockPrint() #Disable printing

            params, params_covariance = self.minimize_fun(method=method_min,max_nfev = int(maxfev))
    
            x_values = og_time #OG time values
            
            if self.Input_Type=='iCovg_iRates':
                kin_output = self.kinetic_output_iCovg_iRates(x_values, *params) #kinetic output from running predicted rate parameters; This kinetic output differs based on what kind of input was given
                kin_output_covg = kin_output[:np.size(og_covg)] #extracting only coverages and not rates etc
                covg_fit= kin_output_covg.reshape(np.shape(og_covg)) #Reshaping to allow for ease in plotting
            # converg = np.sqrt(np.diag(params_covariance))
            enablePrint() #Re-enable printing

        elif option=='cmin':
            
            print("-"*50)
            print('Performing fitting using LMFIT package:')
            print("-"*50)
            print('-Using Method:',method_cmin)
            blockPrint() #Disable printing

            params, params_covariance = self.minimize_custom_fun(method=method_cmin,max_nfev = int(maxfev))
    
            x_values = og_time #OG time values
            
            yfit = self.covg_func(x_values, *params)
            covg_fit=yfit.reshape(np.shape(og_covg))
            #onverg = np.sqrt(np.diag(params_covariance))
            enablePrint() #Re-enable printing

        elif option=='mfit':
            
            print("-"*50)
            print('Performing fitting using LMFIT package:')
            print("-"*50)
            print('-Using Method:',method_fit)
            blockPrint() #Disable printing

            params = self.minimize_fit(method=method_fit, weights_ = weights)
    
            x_values = og_time #OG time values
            
            yfit = self.covg_func(x_values, *params)
            covg_fit=yfit.reshape(np.shape(og_covg[:,1:]))
            #onverg = np.sqrt(np.diag(params_covariance))
            enablePrint() #Re-enable printing
            
        elif option=='ML':
            
            print("-"*50)
            print('Performing fitting using scikit machine learning algorithms:')
            print("-"*50)
            blockPrint() #Disable printing
            covg_inp = og_covg#Input coverage
            n=int(n)
            result=self.ML_model_predict(covg_inp,n,filename,mdl)
            params=result
            
            x_values = og_time #OG time values
            yfit = self.covg_func(x_values, *params)
            covg_fit=yfit.reshape(np.shape(og_covg))
            enablePrint() #Re-enable printing
    
        time = og_time   #Normalized OG time values 
        covg_og = og_covg #Normalized OG coverages
        n = len(self.k)     #Normalized MKM fitted coverages
        
        blockPrint()
        # normalized_data_inp = np.insert(covg_og,0,time,axis=1)
        # normalized_data_MKM = np.insert(covg_fit,0,time,axis=1)
        
        # denormalized_data_inp = self.denormalize(Ext_inp_denorm=normalized_data_inp)
        # denormalized_data_MKM = self.denormalize(Ext_inp_denorm=normalized_data_MKM)
        
        # time_d = denormalized_data_inp[:,0] #OG time values 
        # covg_og_d = denormalized_data_inp[:,1:] #OG coverages
        # covg_fit_d = denormalized_data_MKM[:,1:] #MKM fitted coverages
        enablePrint()
        
        #####Printing out the INITIAL RATE COEFFICIENT GUESS
        print('\n \033[1m' + 'Initial guess: \n'+ '\033[0m')
        print('-> Rate Constants:\n',self.k)
        # if self.CovgDep==True:
        #     for i in np.arange(colmn):
        #         print('-> %s constants:\n'%(str(index[i])),self.Coeff[:,i])
        
        #####Printing out the PREDICTED RATE COEFFICIENTS
        print('\n \033[1m' + 'Final predictions: \n'+ '\033[0m')
        print('-> Rate Constants:\n',params[0:n])
        # if self.CovgDep==True:
        #     for i in np.arange(colmn):
        #         print('-> %s constants:\n'%(str(index[i])),params[(i+1)*n:(i+2)*n])
                
        #####Printing out the CONFIDENCE INTERVALS
        # print('\n \033[1m' + 'Confidence Intervals: \n'+ '\033[0m')
        # print('-> Rate Constants:\n',converg[0:n])
        # if self.CovgDep==True:
        #     for i in np.arange(colmn):
        #         print('-> %s constants:\n'%(str(index[i])),converg[(i+1)*n:(i+2)*n])
        
        self.fitted_k = params #Allowing for the fitted parameters to be extracted globally
        
        
        # Plotting 
        plot_title = 'Fitting rate parameters'
        if plot==False:
            return time,covg_og,covg_fit 
        elif plot==True:
            self.plotting(time,covg_og,covg_fit,self.label,title=plot_title) #Plotting coverage-fits
            return time,covg_og,covg_fit 
    #------------------------------------------------------------------------------------------------------------------------------
    #Function responsible for plotting
    #------------------------------------------------------------------------------------------------------------------------------    
    def plotting(self,time,covg_og,covg_fit,label,title='Fitting rate parameters'):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
            
        for i in np.arange(len(covg_og[0,:])):
            ax.plot(time, covg_og[:,i],'o')
            lbl_og = self.Atomic.columns.values[1+len(self.P):]
 
        for i in np.arange(len(covg_fit[0,:])):
            ax.plot(time, covg_fit[:,i],'-')
            lbl_fit=[]
            for i in np.arange(len(lbl_og)):
                lbl_fit.append(lbl_og[i]+' (fit)')
                
            ax.set_xlabel('Time, t, [s]')
            ax.set_ylabel(r"Coverage, $\theta_i, [ML]$")
            ax.set_title(title)
            
        ax.legend(np.append(lbl_og,lbl_fit),fontsize=10, loc='upper right',facecolor='white', edgecolor ='black', framealpha=1)
        
        
        
