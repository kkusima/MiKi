import sys, os, glob
import numpy as np   #package for numerical arithmetics and analysis
import pandas as pd  #package for dataframe and file extraction/creation
import string        #package to allow for access to alphabet strings
import math          #package to allow for the use of mathematical operators like permutation calculation
from mpmath import * #package for precision control
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
        
        self.dplace,self.rtol,self.atol = self.ODE_Tolerances(Dplace=None,reltol=None,abstol=None) #Controls decimal places - used for mp.dps in mpmath precision control #Specifying ODE Tolerance values

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
    def ODE_Tolerances(self,Dplace=None,reltol=None,abstol=None):
        if Dplace==None:
            Dplace = 50
            self.dplace = Dplace
        else:
            self.dplace = Dplace

        if reltol==None:
            reltol = 1e-8
            self.rtol = reltol
        else:
            self.rtol = reltol

        if abstol==None:
            abstol = 1e-8
            self.atol = abstol
        else:
            self.atol = abstol

        return self.dplace,self.rtol,self.atol
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
        mp.dps= self.dplace
        
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
                #Changing the number of decimal places/precision of input Pressures
                for i in np.arange(len(Pr)):
                    Pr[i]=mpf(Pr[i])
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
                
                dsum += float(self.Stoich.iloc[j,i+1])*r[j] #Calculating the rate of production of a species i
            
            D.append(dsum)    
        
        D = np.transpose(D)    
        if coverage==True:    
            return D[len(self.P):]
        else:
            return D
    #------------------------------------------------------------------------------------------------------------------------------      
    def solve_coverage(self,t=[],initial_cov=[],method='BDF',Tf_eval=[],full_output=False,plot=False): #Function used for calculating (and plotting) single state transient coverages
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

        reltol=self.rtol
        abstol=self.atol    
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
    def get_X_RC_SS(self,p_inc=0.1,k_o_inp=[],rxn=-1):
        #p_inc is the percent increase of rate const. #k_o_inp, is the inputed rate constants to allow for their initial values to be user defined, #rxn is the reaction producing the products being investigated : -1 indicates the last elementary step
        if k_o_inp!=[]:
            k_o = k_o_inp
        else:
            k_o = np.array(self.kextract()) #From Param file
            
        Xrc = [] #Initializing the empty array of degrees of rate control
        self.k = np.array(k_o)
        rin = np.array(self.get_SS_rates_reaction())
        # enablePrint()
        # print('--initial SSrates')
        # print(rin)
        # print('\n')
        
        if rxn>len(rin) or rxn<(-len(rin)):
            raise Exception('An invalid rxn value has been entered')
        else:
            ro = rin[rxn] 
        
        for i in np.arange(len(rin)):
            n = 2*i
            # print(i)
            # print('before:')
            # print(k_o)
            kfwd = k_o[n]*(1+p_inc) #Multiplying the relevant forward rate const. by the change
            krvs = k_o[n+1]*(1+p_inc) #Multiplying the relevant reverse rate const. by the change
            indices = [n,n+1] #The relevant indices in the rate const. array corresponding to this change
            repl = [kfwd,krvs] #The changed rate const.s corresponding to the indices
            knew = np.array(k_o) #Re-initializing knew so it can only be changed for the specific elementary steps (i.e so that other rate constants remain unchanged)
            for index, replacement in zip(indices, repl):
                knew[index] = replacement
            # print('after')
            # print(knew)
            self.k = np.array(knew)
            rnew = np.array(self.get_SS_rates_reaction())
            # print('\n Printing rnew')
            # print(rnew)
            Xrc.append((rnew[rxn]-ro)/(ro*p_inc))
        
        self.k = np.array(k_o)
        # print(self.k)    
        # blockPrint()
                        
        return Xrc
    
    #------------------------------------------------------------------------------------------------------------------------------
    #Functions to calculate the resulting kinteic parametes from pressure switching
    #------------------------------------------------------------------------------------------------------------------------------        
    def periodic_operation_two_states(self,State1=[],State2=[],t1=None,t2=None,total_time=None,n_cycles=None,Initial_Covg=[],label='coverages',plot=False): #Function used for calculating (and plotting) the dynamic transient coverages
        if t1==None or t2==None:
            raise Exception("t1, t2 must be inputted and either total_time or n_cycles should be inputed")
        if total_time!=None and n_cycles!=None:
            raise Exception(" Can not enter both total_time and n_cycles")
        if total_time==None and n_cycles!=None:
            total_time = (t1+t2)*n_cycles

        #If State conditions are not inputted - forcing the user to input them    
        if State1==[]:
            print('\nThe Pressure Input Format:P1,P2,P3,...\n')
            print("Enter the Pressure Conditions of State 1 below:")
            State1_string = input().split(',')
            State1 = [float(x) for x in State1_string]
        if State2==[]:  
            print("Enter the Pressure Conditions of State 2 below:")
            State2_string = input().split(',')
            State2 = [float(x) for x in State2_string]


        States = ['State 1','State 2']
        total_time_f=0 #final full simulation time
        new_ti = 0 #Initial time
        new_tf = t1 #Staring end time for state1
        current_state = "State 1"  #starting state

        full_feature=pd.DataFrame()  #Initializing full vector of coverages
        full_time=[0.0] #Intiialising full vector of times
        States_Table=pd.DataFrame() #Initialize Table for demonstrating states

        if Initial_Covg==[]:
            Initial_Covg = self.init_cov

        while (total_time_f<total_time):
            blockPrint() #Disable printing (since warning will show up if steady state not reached)

            #Setting the pressures and automatically switching states
            if current_state == "State 1":
                self.set_rxnconditions(Pr=State1)
                current_state = "State 2"
            elif current_state == "State 2":
                self.set_rxnconditions(Pr=State2)
                current_state = "State 1"

            self.Ti = new_ti
            self.Tf = new_tf

            self.label = label
            if self.label == 'coverages':
                solb,soltb=self.solve_coverage(initial_cov=Initial_Covg)
            elif self.label == 'rates_p':
                solb,soltb=self.solve_rate_production(initial_coverage=Initial_Covg)
            elif self.label == 'rates_r':
                solb,soltb=self.solve_rate_reaction(initial_coverage=Initial_Covg)

            soltb = soltb-soltb[0]

            #Updating Simulation Times
            new_ti = soltb[-1]
            new_tf = new_ti+t2

            Initial_Covg = self.get_SS_coverages()

            full_feature = pd.concat([full_feature,pd.DataFrame(solb)], axis=0)
            full_time=np.hstack([full_time,full_time[-1]+soltb])
            
            total_time_f=full_time[-1]

            enablePrint() #re-enabling printing

        full_feature = full_feature.to_numpy()
        full_time = full_time[1:]

        if self.label == 'coverages':
            title = 'coverages'
        elif self.label == 'rates_p':
            title = 'Rates of production'
        elif self.label == 'rates_r':
            title = 'Rates of reaction'

        print('\nPeriodic Simulation of',title,'\n' )

        States_Table = pd.concat([States_Table,pd.DataFrame(self.Atomic.columns.values[1:])[:len(self.P)]], axis=0)
        States_Table[States[0]+',P[bar]'] = pd.DataFrame([np.format_float_scientific(i, exp_digits=2) for i in State1])
        States_Table[States[1]+',P[bar]'] = pd.DataFrame([np.format_float_scientific(i, exp_digits=2) for i in State2])

        enablePrint()
        print(States_Table)
        print('\nNumber of Cycles:', total_time/(t1+t2) , '\n')
        
        if plot==False:
            return full_feature,full_time
        elif plot==True:
            self.label = label
            self.plotting(full_feature,full_time,self.label)
            return full_feature,full_time
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
        Ngs = len(self.Pextract())
        Ncs = len(self.Stoich.iloc[0,:])-Ngs-1 #No. of Surface species
        if Source=='Model':
            if Param=='Pressure':
                enablePrint()
                print('\n Order for Input Pressures [Pa]:')
                Pr_header = np.array(self.Stoich.columns[1:Ngs+1])
                x = []
                for i in np.arange(len(Pr_header)): x.append(i)
                list = [x]
                print(pd.DataFrame(list,columns=Pr_header, index=['Array order']))
                return 

            elif Param=='Coverage':
                enablePrint()
                print('\n Order for Input Coverages (Transient and Steady State) [ML]:')
                Covg_header = np.array(self.Stoich.columns[Ngs+1:])
                x = []
                for i in np.arange(len(Covg_header)): x.append(i)
                list = [x]
                print(pd.DataFrame(list,columns=Covg_header, index=['Array order']))
                return 

            elif Param=='Rates_Production':
                enablePrint()
                print('\n Order for Input Rates of Production (Transient and Steady State) [TOF]:')
                Rp_header = np.array(self.Stoich.columns[1:])
                x = []
                for i in np.arange(len(Rp_header)): x.append(i)
                list = [x]
                print(pd.DataFrame(list,columns=Rp_header, index=['Array order']))
                return 

            elif Param=='Rates_Reaction':
                enablePrint()
                print('\n Order for Input Rates of Reactions (Transient and Steady State) [TOF]:')
                Rr_header = np.array(self.Stoich.iloc[:,0])
                x = []
                for i in np.arange(len(Rr_header)): x.append(i)
                list = [x]
                print(pd.DataFrame(list,columns=Rr_header, index=['Array order']))
                return Param

            elif Param=='Rate_Constants':
                enablePrint()
                print('\n Order for Input Rate Constants [1/s]:')
                params_header = []
                for i in np.arange(len(self.k)):
                    params_header.append('k'+str(i+1))
                x=[]
                for i in np.arange(len(params_header)): x.append(i)
                list = [x]
                print(pd.DataFrame(list,columns=params_header, index=['Array order']))
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
        
#-----------------------------------------------------------------------------------------------------------------------------------------        
class ML_Fitting:    
    def __init__(self,Atomic_csv,Stoich_csv,Param_Guess_csv,Input_csv,Sim_folder_name='Sim_*',Input_Format ='KMC'): #Inputs necessary to initialize the MK Model
        self.MKM = MKModel(Atomic_csv,Stoich_csv,Param_Guess_csv) #Initializing the MF-MK Model
        self.MKM_test_inp = MKModel(Atomic_csv,Stoich_csv,Param_Guess_csv)

        self.dplace,self.rtol,self.atol = self.MKM.ODE_Tolerances() #Controls decimal places - used for mp.dps in mpmath precision control #Specifying ODE Tolerance values
        
        
        self.Input = pd.read_csv(Input_csv)
        self.Atomic = pd.read_csv(Atomic_csv)     #Opening/Reading the Atomic input file needed to be read
        self.Stoich = pd.read_csv(Stoich_csv)    #Opening/Reading the Stoichiometric input file needed to be read
        self.Param_Guess = pd.read_csv(Param_Guess_csv)     #Opening/Reading the Parameter of guess input file needed to be read       

        self.fit_k = False

        self.forced_param = self.MKM.k
        self.fitted_param = self.MKM.k
        
        self.test_train_split = 0.1
        self.ML_algorithm = 'KNN'
       
        self.Input_Format = Input_Format

        if self.Input_Format == 'KMC':
            print('Type of Experimental Training Files: KMC')
            self.Input_KMC_init = self.KMC_Input_Folders_Initialization()
            self.EXP_Dictionary = self.KMC_Dictionary()

        # self.X_train = X_train
        # self.Y_train = Y_train
        # self.X_test = X_test
        # self.Y_test = Y_test
        # self.Exp_gas_name
        # self.Exp_surf_name
        # self.KMC_n_gas_species
        # self.KMC_n_surf_species
        # self.External_Dataframe() #External Input From Input CSV, to be fitted on
        # self.external_data_input
        # self.external_data_output

        
        # self.P ,self.Temp = self.set_rxnconditions() #Setting reaction conditions (defaulted to values from the Param File but can also be set mannually )
        # self.Ti,self.Tf=self.set_limits_of_integration() #Sets the range of time needed to solve for the relavant MK ODEs, defaults to 0-6e6 but can also be manually set
        # self.rate_const_correction = 'None' #Accounting for correction to the rate constants (i.e. enhancing the mean field approximation)
        # self.MKM.rate_const_correction = self.rate_const_correction
        # self.Input_Type = Input_Type
        # self.PARAM = self.set_params
        # self.BG_matrix='auto' #Bragg williams constant matrix
        # self.Coeff = self.Coeff_extract() #Extracting the coverage dependance coefficients
        # self.init_cov=self.set_initial_coverages() #Sets the initial coverage of the surface species, defaults to zero coverage but can also be set manually
        # self.n_extract = 0.5 #Nummber of points to be extracted from the input file #Defaulted to 0.5 of the total points
        # self.status='Waiting' #Used to observe the status of the ODE Convergence
        # self.label='None'   #Used to pass in a label so as to know what kind of figure to plot
        # #Output: self.fitted_k  #Can be used to extracted an array of final fitted rate parameters
        # if self.Input_Type not in ['iCovg','iCovg_iRates']:
        #     raise Exception('Input type specified is not recognised.\n Please make sure your input type is among that which is acceptable')   
    #-----------------------------------------------------------------------------------------------------------------------------------------
    def KMC_Input_Folders_Initialization(self):
        os.getcwd()

        Sim_folder_names = []
        i = 0
        for file in glob.glob("Sim_*"):
            Sim_folder_names.append(file)
            i+=1
        
        self.KMC_Sim_folder_names = Sim_folder_names
        
        #--------------------------------------------------------------------------------------------
        
        set_init_coverages = np.empty([len(Sim_folder_names),4])
        #Remember: A='CO*'; B='O*'
        #Reading A and B initial coverages from the KMC simulation input coverage files
        c = 0 #counter
        for s in Sim_folder_names:
            set_coverages = []
            for i in np.arange(len(s)):
                if i<(len(s)-2) and s[i].isdigit() and (s[i+1]).isdigit() and (s[i+2]).isdigit():
                    cov_triple = int(s[i:i+3])
                    set_coverages.append(cov_triple)
                    
                elif i<(len(s)-1) and s[i].isdigit() and (s[i+1]).isdigit()and not((s[i-1]).isdigit()):
                    cov_double = int(s[i:i+2])
                    set_coverages.append(cov_double)
                    
                elif s[i].isdigit() and not((s[i-1]).isdigit()) and not((s[i-2]).isdigit()):
                    cov_single = int(s[i])
                    set_coverages.append(cov_single)
                                        #B_O*_covg,     A_CO*_covg,     O2*_covg,*_covg
            set_init_coverages[c,:] = [set_coverages[1],set_coverages[0],0,100-sum(set_coverages)]
            c+=1 #counter
        
        os.getcwd()
        self.KMC_set_init_coverages = set_init_coverages
        #----------------------------------------------------------------------------------------------------

        #Checking to see match
        ## Copying all the other input files into the different simulation folders
        # Extracting initial coverages
        #TO BE SET AS INPUTS
        #Remember: A='CO*'; B='O*'
        n_points = 500 #From KMC simulation 
        n_gas_species = 3 #From KMC simulation
        n_surf_species = 4 #From KMC simulation
        self.KMC_n_points = n_points
        self.KMC_n_gas_species = n_gas_species
        self.KMC_n_surf_species = n_surf_species

        Exp_init_coverages = np.empty([len(Sim_folder_names),n_surf_species])
        c = 0 #counter
        for s in Sim_folder_names:
            os.chdir(s)
            file=open('specnum_output.txt','r').readlines() #Reading in the relevant file
            b=[]
            for i in np.arange(len(file)): 
                b.append(file[i].split())                   #Dividing the rows into columns
            o = pd.DataFrame(data=b)                        #Final output

        #     print(o)
            #Extracting Number of Sites from the general_output file:
            inp=open('general_output.txt','r').readlines()
            for i in np.arange(len(inp)): 
                if 'Total number of lattice sites:' in inp[i]:
                    val = i  #Line in text file where sentence is present

            sites = int(inp[val][35:])
            
            #Finding number of surface species
            headings = (o.iloc[0,:])
            n_ss = sum('*' in h for h in headings) #Number of surface species
            
            #Finding number of gas species
            n_gs = len(headings)-5-n_ss
            
            #Adding column to calculate number of empty sites
            n_c=(len(o.iloc[0,:])) #number of current columns
            o[n_c]=" "           #Creating new empty column 
            o.iloc[0,n_c]="*"    #Labelling the new empty column 

            st = 0 #Initializing empty site coverage vector


            for i in range(len(o.iloc[1:])):
                if n_ss==0:
                    site = sites
                else:
                    for j in range(n_ss):
                        st = st + float(o.iloc[i+1,5+j]) #Calculating no. of empty sites #Asuming empty sites are first to be reportes (i.e @5)
                    site = sites - st
                    st = 0
                o.iloc[i+1,n_c] = site
            
            Sspecies = []
            for i in range(n_ss):
                Sspecies.append(5+i) 
            Sspecies.append(len(o.iloc[1,:])-1)#Including empty sites

            #Calculating itme:
            Gtime = o[2][1:].astype(float) 
            #Calculating coverages:
            Scoverages = np.empty([len(o.iloc[:,1])-1,len(Sspecies)])
            for i in range(len(Scoverages[1,:])):
                Scoverages[:,i] = o[Sspecies[i]][1:].astype(float)/sites
                
            exp_init_covg = []
            for i in np.arange(n_surf_species):    #B_O*_covg,     A_CO*_covg,     O2*_covg, *_covg
                exp_init_covg.append(Scoverages[0,i])
                
            Exp_init_coverages[c,:] = exp_init_covg
            
            c+=1
            
            
            os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir)) #Changes directory back to where this script is

        self.KMC_Exp_init_coverages = Exp_init_coverages
        #---------------------------------------------------------------------------------------------------------------

        if np.shape(set_init_coverages) != np.shape(Exp_init_coverages):
            raise Exception('Not all simulations have been completed successfully')
            
        for i in np.arange(np.shape(set_init_coverages)[0]):
            for j in np.arange(np.shape(set_init_coverages)[1]):
                norm_val = set_init_coverages[i,j]
                exp_val = round(Exp_init_coverages[i,j])
                if not( norm_val + 1 > exp_val) and not(norm_val - 1 < exp_val): #i.e if not within range
                    raise Exception('Initial coverages used in the simulation are not the same as it was set')
                    
                if (i==(np.shape(set_init_coverages)[0] - 1) and j==(np.shape(set_init_coverages)[1] - 1)):
                    fin_text = 'SIMULATIONS MATCH AS EXPECTED'


        text = "Relevant Input Files are successfuly Initialized."

        return print('\n','Number of simulations:',i,'\n',text,'\n', fin_text, '\n')
    

#-----------------------------------------------------------------------------------------------------------------------------------------
    def KMC_Dictionary(self):
        n = len(self.KMC_Sim_folder_names)
        self.KMC_n_sim = n
        Covg = np.zeros((n,self.KMC_n_points,self.KMC_n_surf_species)) #O*, CO*, O2*, *
        Rates = np.zeros((n,self.KMC_n_points,self.KMC_n_gas_species)) #O2, CO,  CO2
        KMC_time_Array = np.zeros((n,self.KMC_n_points))
        init_coverages = np.empty([n,self.KMC_n_surf_species])
        c = 0 #counter for number of simulation (folders)

        for s in self.KMC_Sim_folder_names:
            os.chdir(s)
            file=open('specnum_output.txt','r').readlines() #Reading in the relevant file
            b=[]
            for i in np.arange(len(file)): 
                b.append(file[i].split())                   #Dividing the rows into columns
            o = pd.DataFrame(data=b)                        #Final output

        #     print(o)
            #Extracting Number of Sites from the general_output file:
            inp=open('general_output.txt','r').readlines()
            for i in np.arange(len(inp)): 
                if 'Total number of lattice sites:' in inp[i]:
                    val = i  #Line in text file where sentence is present

            sites = int(inp[val][34:])
            
            #Finding number of surface species
            headings = (o.iloc[0,:])
            n_ss = sum('*' in h for h in headings) #Number of surface species
            
            #Finding number of gas species
            n_gs = len(headings)-5-n_ss
            
            #Adding column to calculate number of empty sites
            n_c=(len(o.iloc[0,:])) #number of current columns
            o[n_c]=" "           #Creating new empty column 
            o.iloc[0,n_c]="*"    #Labelling the new empty column 

            st = 0 #Initializing empty site coverage vector


            for i in range(len(o.iloc[1:])):
                if n_ss==0:
                    site = sites
                else:
                    for j in range(n_ss):
                        st = st + float(o.iloc[i+1,5+j]) #Calculating no. of empty sites #Asuming empty sites are first to be reportes (i.e @5)
                    site = sites - st
                    st = 0
                o.iloc[i+1,n_c] = site
            
            Sspecies = []
            for i in range(n_ss):
                Sspecies.append(5+i) 
            Sspecies.append(len(o.iloc[1,:])-1)#Including empty sites
            
            self.Exp_Sspecies = (o.iloc[0,Sspecies].tolist())

            #Calculating itme:
            Gtime = o[2][1:].astype(float) 
            
            #Calculating coverages:
            Scoverages = np.empty([len(o.iloc[:,1])-1,len(Sspecies)])
            for i in range(len(Scoverages[1,:])):
                Scoverages[:,i] = o[Sspecies[i]][1:].astype(float)/sites
                
            Gspecies = []
            for i in range(n_gs):
                Gspecies.append(5+n_ss+i)

            self.Exp_Gspecies = (o.iloc[0,Gspecies].tolist())

            #Extracting the number of gas species molecules:    
            Gnmol = np.empty([len(o.iloc[:,1])-1,len(Gspecies)])
            for i in range(len(Gnmol[1,:])):
                Gnmol[:,i] = o[Gspecies[i]][1:].astype(float)
            
            ### Calculating the instantaneous rates of profuction (i.e grad/sites)
            TOF_GS = np.empty([len(o.iloc[:,1])-1,len(Gspecies)]) #initializing an array of instantaneous TOFs for gaseous species

            for i in np.arange(len(Gspecies)):
                grads = np.gradient(Gnmol[:,i],Gtime,edge_order=2)
                TOF_GS[:,i] = grads/sites
            
            
            #initializing TOF for gas species
            STOF = np.empty([self.KMC_n_points,self.KMC_n_gas_species])
            gs_names = (o.iloc[0,Gspecies].tolist())
            gs_names_colmn = []
            
            for i in np.arange(self.KMC_n_gas_species): #Collecting TOFs
                STOF[:,i] = pd.Series(TOF_GS[:,i])
                
            for i in gs_names: #Collecting gas names
                gs_names_colmn.append('R_'+i)
            
            Rates_p = pd.DataFrame(STOF,
                            columns = gs_names_colmn)

            init_covg = []
            for i in np.arange(self.KMC_n_surf_species):    #B_O*_covg,     A_CO*_covg,     O2*_covg, *_covg
                init_covg.append(Scoverages[0,i])
                
            init_coverages[c,:]= init_covg #Initial coverages
            
            KMC_time_Array[c,:]= Gtime #Time matrix
            
            Covg[c,:,:] = Scoverages #Coverage profile tensor
            
            Rates[c,:,:] = Rates_p
            
            c+=1
            
            os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir)) #Changes directory back to where this script is

        #https://stackoverflow.com/questions/49881570/python-dictionaries-appending-arrays-to-a-dictionary-for-a-specific-key
        data_KMC_dict = {'init_covg': init_coverages, 'sim_time': KMC_time_Array, 'covg_prof': Covg, 'iRates': Rates}
        return data_KMC_dict
    #---------------------------------------------------------------------------------------------------------------
    def Tensor_To_Array(self,Sim_tens):
        a = Sim_tens
        m,n,r = a.shape
        sim_arr = np.column_stack((np.repeat(np.arange(m),n),a.reshape(m*n,-1)))
        return sim_arr
    
    #---------------------------------------------------------------------------------------------------------------
    def Creating_Input_Feature_Tensor(self):
        if self.Input_Format =='KMC':
            ## Creating Simulation file names input----------------------------------------------------------------------------
            Sim_names_tens = np.empty((self.KMC_n_sim,self.KMC_n_points,1),dtype=np.dtype('U100'))
            for i in np.arange(self.KMC_n_sim):
                for j in np.arange(self.KMC_n_points):
                    Sim_names_tens[i,j,:] = self.KMC_Sim_folder_names[i]

            ## Creating Init coverages tensor input----------------------------------------------------------------------------
            ini_covg_tens = np.empty((self.KMC_n_sim,self.KMC_n_points,self.KMC_n_surf_species),dtype=float)
            for i in np.arange(self.KMC_n_sim):
                for j in np.arange(self.KMC_n_points):
                    ini_covg_tens[i,j,:] = self.EXP_Dictionary['init_covg'][i,:]

            ## Creating time tensor input----------------------------------------------------------------------------
            sim_time_tens = np.empty((self.KMC_n_sim,self.KMC_n_points,1),dtype=float)
            for i in np.arange(self.KMC_n_sim):
                for z in np.arange(1):
                    sim_time_tens[i,:,z] = self.EXP_Dictionary['sim_time'][i,:]

        Inp_feature_tensors = {'Sim_names_tens': Sim_names_tens, 'ini_covg_tens': ini_covg_tens, 'sim_time_tens': sim_time_tens}
        return Inp_feature_tensors
    #-------------------------------------------------------------------------------------------------------------------------------------
    def Exp_Dataframe(self):

        self.Inp_feature_tensors = self.Creating_Input_Feature_Tensor()

        exp_df = pd.DataFrame(self.Tensor_To_Array(self.Inp_feature_tensors['Sim_names_tens']),columns= ['Sim_ndex','Sim_names'])

        if self.Input_Format=='KMC':
            #Adding initial coverages
            surf_names = self.Exp_Sspecies
            for i in np.arange(self.KMC_n_surf_species):
                spec = surf_names[i]
                exp_df['Init_Covg_'+spec] = pd.DataFrame(self.Tensor_To_Array(self.Inp_feature_tensors['ini_covg_tens']))[1+i]

            #Adding time
            exp_df['Time'] = pd.DataFrame(self.Tensor_To_Array(self.Inp_feature_tensors['sim_time_tens']))[1]

            #Adding coverage profiles of surface species
            surf_names = self.Exp_Sspecies
            for i in np.arange(self.KMC_n_surf_species):
                spec = surf_names[i]
                exp_df['KMC_Covg_'+spec] = pd.DataFrame(self.Tensor_To_Array(self.EXP_Dictionary['covg_prof']))[1+i]

            #Adding iRates profiles of gaseous species
            gs_names = self.Exp_Gspecies
            for i in np.arange(self.KMC_n_gas_species):
                spec = gs_names[i]
                exp_df['KMC_iRates_'+spec] = pd.DataFrame(self.Tensor_To_Array(self.EXP_Dictionary['iRates']))[1+i]

        return exp_df
    #-------------------------------------------------------------------------------------------------------------------------------------
    def MKModelling(self,*fit_params):

        if self.Input_Format == 'KMC': #need to generalize init_covgs more
            MKM_init_coverages = np.empty([len(self.KMC_Sim_folder_names),self.KMC_n_surf_species])

            self.MKM.ODE_Tolerances(Dplace=50,reltol=1e-8,abstol=1e-8)

            n = self.KMC_n_sim
            n_points = self.KMC_n_points #From KMC simulation 
            n_gas_species = self.KMC_n_gas_species #From KMC simulation
            n_surf_species = self.KMC_n_surf_species #From KMC simulation
            MKM_Covg = np.zeros((n,n_points,n_surf_species)) # O*,CO*, O2*, * #Make sure KMC order of species matches MKM inputs
            MKM_Rates = np.zeros((n,n_points,n_gas_species)) #O2, CO, CO2     #Make sure KMC order of species matches MKM inputs
            time_MKM_Array = np.zeros((n,n_points))

            self.MKM.set_limits_of_integration(Ti=self.EXP_Dictionary['sim_time'][0][0],Tf=self.EXP_Dictionary['sim_time'][-1][-1])
            
            self.MKM.k = np.array(fit_params)
            
            #Remember: A='CO*'; B='O*'
            #Reading A and B initial coverages from the KMC simulation input coverage file names!
            c = 0 #counter
            for s in self.KMC_Sim_folder_names:
                set_coverages = []
                for i in np.arange(len(s)):
                    if i<(len(s)-2) and s[i].isdigit() and (s[i+1]).isdigit() and (s[i+2]).isdigit():
                        cov_triple = int(s[i:i+3])
                        set_coverages.append(cov_triple)

                    elif i<(len(s)-1) and s[i].isdigit() and (s[i+1]).isdigit()and not((s[i-1]).isdigit()):
                        cov_double = int(s[i:i+2])
                        set_coverages.append(cov_double)

                    elif s[i].isdigit() and not((s[i-1]).isdigit()) and not((s[i-2]).isdigit()):
                        cov_single = int(s[i])
                        set_coverages.append(cov_single)
                                            #B_O*_covg,     A_CO*_covg,     O2*_covg,*_covg  #Note: Special case: Simulation naming switches from KMC and MKM order
                init_covgs = [set_coverages[1]/100,set_coverages[0]/100,0,(100-sum(set_coverages))/100]
                
                self.MKM.set_initial_coverages(init=init_covgs)
                MKM_init_coverages[c,:] = [float(i) for i in init_covgs]
                
                sola,solta = self.MKM.solve_coverage(Tf_eval=self.EXP_Dictionary['sim_time'][0],plot=False)
                time_MKM_Array[c,:]= solta #Time matrix
                MKM_Covg[c,:,:] = sola #Coverage profile tensor

                solb,soltb = self.MKM.solve_rate_production(Tf_eval=self.EXP_Dictionary['sim_time'][0],plot=False)
                MKM_Rates[c,:,:] = solb[:,0:n_gas_species] 

                c+=1 #counter
            return {'init_covg': MKM_init_coverages, 'sim_time': time_MKM_Array, 'covg_prof': MKM_Covg, 'iRates': MKM_Rates}
    #-------------------------------------------------------------------------------------------------------------------------------------
    def MKM_k_fitting(self,*fit_params,feature = 'iRates'):
        data_MKM_dict  = self.MKModelling(*fit_params)    
        return np.reshape(data_MKM_dict[feature],data_MKM_dict[feature].size)
    #-------------------------------------------------------------------------------------------------------------------------------------
    #def k_opt_fitting(self):
    # x_values = data_KMC_dict['sim_time'] #Normalized Input Time variables (Independent Variable) (eg. KMC Time)
    # y_values = np.reshape(data_KMC_dict['iRates'],data_KMC_dict['iRates'].size) #Normalized Input Dependent variable(s) (eg. KMC coverages)

    # initial_vals = np.array(MKM.k)

    # params, params_covariance = optimize.curve_fit(MKM_k_fitting, x_values, y_values
    #                                             ,method = 'trf', bounds=(0,1e10), maxfev=1e3, xtol=1e3, ftol=1e3
    #                                             ,p0=initial_vals)

    #-------------------------------------------------------------------------------------------------------------------------------------
    def Exp_MKM_Dataframe(self,params=[]): #Adding MKM Data
        self.MKM.ODE_Tolerances(Dplace=50,reltol=1e-5,abstol=1e-8)
        if params==[]:
            params = self.MKM.k
        else:
            self.MKM.k = params
            self.MKM_test_inp.k = params


        if self.fit_k == False:
            self.MKM_Dictionary = self.MKModelling(*params)
        # elif self.fit_k == True:
        #     self.MKM_Dictionary = k_opt_fitting

        exp_mkm_df = self.Exp_Dataframe()

        if self.Input_Format == 'KMC':
            
            #Adding coverage profiles of surface species
            surf_names = self.Exp_Sspecies
            for i in np.arange(self.KMC_n_surf_species):
                spec = surf_names[i]
                exp_mkm_df['MKM_Covg_'+spec] = pd.DataFrame(self.Tensor_To_Array(self.MKM_Dictionary['covg_prof']))[1+i]


            #Adding iRates profiles of gaseous species
            gs_names = self.Exp_Gspecies
            for i in np.arange(self.KMC_n_gas_species):
                spec = gs_names[i]
                exp_mkm_df['MKM_iRates_'+spec] = pd.DataFrame(self.Tensor_To_Array(self.MKM_Dictionary['iRates']))[1+i]

        return exp_mkm_df
    #-------------------------------------------------------------------------------------------------------------------------------------
    def Fit_Evaluation_RMSE(self, params=[]):
        from sklearn.metrics import mean_squared_error
        if params==[]:
            params = self.fitted_param
        else:
            self.MKM.k = params
            self.MKM_test_inp.k = params

        out_df = self.Exp_MKM_Dataframe(params = params)
        rmse_matrix = []
        for i in np.arange(len(set(out_df['Sim_ndex']))):
            
            df = out_df.loc[out_df['Sim_ndex'] == str(i)]
            
            #calculating covg  ---------------------------------------------------------------------------------
            df = out_df.loc[out_df['Sim_ndex'] == str(i)] #Extracting dataframe only corresponding to simulation i
            kmc_dat_covg = df[[col for col in df if 'KMC_Covg' in col]] #Extracting KMC comp data 
            mkm_dat_covg = df[[col for col in df if 'MKM_Covg' in col]] #Extracting MKM comp data 
            
            ls = kmc_dat_covg.columns.to_list()
            covg_nm = [string[3:] for string in ls] #surface_species names
            
            rmse_covg = []
            for i in np.arange(len(covg_nm)):
                rmse_covg.append(sqrt(mean_squared_error(kmc_dat_covg['KMC'+covg_nm[i]], mkm_dat_covg['MKM'+covg_nm[i]])))
                
            
            #calculating irates ---------------------------------------------------------------------------------
            
            kmc_dat_irates = df[[col for col in df if 'KMC_iRates' in col]] #Extracting KMC comp data 
            mkm_dat_irates = df[[col for col in df if 'MKM_iRates' in col]] #Extracting MKM comp data 
            
            ls = kmc_dat_irates.columns.to_list()
            irates_nm = [string[3:] for string in ls] #gas_species names
            
            rmse_irates = []
            for i in np.arange(len(irates_nm)):
                rmse_irates.append(sqrt(mean_squared_error(kmc_dat_irates['KMC'+irates_nm[i]], mkm_dat_irates['MKM'+irates_nm[i]])))
                
            rmse_matrix.append(rmse_covg+rmse_irates)
            
            rmse_names = covg_nm+irates_nm
            
        #Creating the RMSE Dataframe

        RMSE_Dataframe = pd.DataFrame(list(set(out_df['Sim_names'])), columns = ['Sim_names'])

        for i in np.arange(len(rmse_names)):
            spec = rmse_names[i]
            RMSE_Dataframe['RMSE'+spec] = pd.DataFrame(rmse_matrix).applymap(lambda x: round(x, 3))[i]

        return RMSE_Dataframe
    #-------------------------------------------------------------------------------------------------------------------------------------
    def Fit_Visualization(self,params = [],Comp='Covg'): #plots all Covg or iRates plots 

        if params==[]:
            params = self.MKM.k
        else:
            self.MKM.k = params
            self.MKM_test_inp.k = params

        exp_mkm_df = self.Exp_MKM_Dataframe(params = params)
        out_df = exp_mkm_df
        

        if self.Input_Format =='KMC':
            print('Comparison of KMC vs fitted-k MKM results for' + Comp)

            #Part 2: Plot comparison results for fitting analysis
            for i in np.arange(len(set(out_df['Sim_ndex']))): #For each simulation:
                    #Extracting KMC results: ------------------------------------------------
                    df = out_df.loc[out_df['Sim_ndex'] == str(i)] #Extracting dataframe only corresponding to simulation i
                    kmc_dat = df[[col for col in df if 'KMC_'+Comp in col]].to_numpy() #Extracting KMC comp data as array
                    Time = df['Time'].to_numpy()

                    #Plotting KMC result : ------------------------------------------------          
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    
                    for i in np.arange(len(kmc_dat[0,:])):
                        ax.plot(Time, kmc_dat[:,i],'--')
                                    
                    if Comp =='iRates':
                        leg_nd = self.Exp_Gspecies
                        ax.set_ylim([-0.2,0.2])
                    elif Comp == 'Covg':
                        leg_nd = self.Exp_Sspecies
                        
                    ax.set_xlabel('Time, t, [s]')
                    if Comp =='iRates':
                        ax.set_ylabel(r"Rates of Production, $R_i$")
                        ax.set_title('Rates of production versus Time_ for Simulation_'+ df['Sim_ndex'].iloc[i] +': _'+df['Sim_names'].iloc[0]+'| A:O* ; B:CO*')
                    elif Comp == 'Covg':
                        ax.set_ylabel(r"Coverage, $\theta_i, [ML]$")
                        ax.set_title('Coverages versus Time_for Simulation_'+ df['Sim_ndex'].iloc[i] +': _'+df['Sim_names'].iloc[0]+'| A:O* ; B:CO*')
                    
                    
                    #Extracting MKM results: ------------------------------------------------
                    mkm_dat = df[[col for col in df if 'MKM_'+Comp in col]].to_numpy() #Extracting MKM comp data as array
                    Time = df['Time'].to_numpy()
                        
                        
                    #Adding to the plot, MKM result : ------------------------------------------------     
                    for i in np.arange(len(mkm_dat[0,:])):
                        ax.plot(Time, mkm_dat[:,i])
                    
                    #Plotting all the legends together
                    ax.legend([f"{string}_KMC" for string in leg_nd]+[f"{string}_MKM" for string in leg_nd],fontsize=10, loc='upper right',facecolor='white', edgecolor ='black', framealpha=1)


    #-------------------------------------------------------------------------------------------------------------------------------------
    def Vis_Exp_MKM_Dataframe(self,p_c = 0.1):

        exp_mkm_df = self.Exp_MKM_Dataframe()
        out_df = exp_mkm_df

        import random
        print('Percent of test data selected:',p_c*100,'%')


        max_sim_number = int(len(set(out_df['Sim_ndex']))) #MAx_number of simulations present #Count is starting from 0
        n_test_sim = int(p_c*max_sim_number) #Number of simulations being used as test
        sim_nums = list(set(out_df['Sim_ndex'])) # List of unique simulation numbers

        test_sims = random.sample(sim_nums,n_test_sim) #Random sim_numbers for testing

        print('\n','The list of simulations used in the test dataset:\n',test_sims)

        Vis_Exp_MKM_ = out_df.loc[out_df['Sim_ndex'].isin(test_sims)]
        return Vis_Exp_MKM_
    
    #-------------------------------------------------------------------------------------------------------------------------------------
    def three_d_plot_Exp_only(self,p_c = 0.1,Comp='Covg'):
        ##Constructing a waterfall plot
        from matplotlib.collections import PolyCollection
        from matplotlib.collections import LineCollection
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib import colors as mcolors
        import numpy as np
        import matplotlib

        New_out_df = self.Vis_Exp_MKM_Dataframe(p_c = p_c)


        if self.Input_Format=='KMC':
            exp_name ='KMC'
            plt.figure(1)
            axes=plt.axes(projection="3d")

            def colors(arg):
                return mcolors.to_rgba(arg, alpha=6)

            x1 = New_out_df.loc[New_out_df['Sim_ndex'] == New_out_df.iloc[0][0]]['Time'].to_numpy()  #Sim_ndex = fist sim on the list and then find array of corresponding time
            verts1 = []
            verts2= []
            Sim_len = len(set(New_out_df['Sim_ndex']))
            sims = np.array(list(set(New_out_df['Sim_ndex'])))
            z1 = np.arange(Sim_len)
            for z in z1:
                df = New_out_df.loc[New_out_df['Sim_ndex'] == str(sims[z])] #Extracting only the dataframe corrsponding to simulation z
                y1 = df[[col for col in df if exp_name+'_'+Comp+'_O*' in col]].to_numpy()
                y2 = df[[col for col in df if exp_name+'_'+Comp+'_CO*' in col]].to_numpy()

                verts1.append(list(zip(x1, y1)))
                verts2.append(list(zip(x1, y2)))
                
            facecolors = [matplotlib.cm.jet(x) for x in np.random.rand(Sim_len)]

            poly1 = LineCollection(verts1,color = facecolors,linewidths=(1,),zorder=2,linestyle='-')
            poly2 = LineCollection(verts2,color = facecolors,linewidths=(1,),zorder=2,linestyle='--')

            # Removes shaded region
            poly1.set_facecolor(None)
            poly2.set_facecolor(None)


            poly1.set_alpha(0.6)
            poly2.set_alpha(0.6)
            axes.add_collection3d(poly1, zs=z1, zdir='y')
            axes.add_collection3d(poly2, zs=z1, zdir='y')

            axes.set_xlabel('X : Time')
            axes.set_xlim3d(0, x1[-1])
            axes.set_ylabel('Y : Simulation')
            axes.set_ylim3d(0, Sim_len,auto=False)
            # axes.yticks(sims.astype(float))
            # axes.set_yticks(sims.astype(float))
            axes.set_yticklabels(np.array(list(set(New_out_df['Sim_names']))),fontdict={'fontsize': 6,'fontweight': 10,'verticalalignment': 'baseline'})
            axes.set_zlabel(Comp)
            # axes.set_zlim3d(0, 1)
            # axes.set_zlim3d(-0.2, 0.2)
            axes.set_title(exp_name+'_Results')

            line_1 = Line2D([0], [0], color='black', linewidth=0.6, linestyle='-') 
            line_2 = Line2D([0], [0], color='black', linewidth=0.6, linestyle='--')

            label_1 = exp_name+'_'+'Covg_O*'
            label_2 = exp_name+'_'+'Covg_CO*'

            lines = [line_1,line_2]
            labels = [label_1,label_2]
            axes.legend(lines, labels, title = "A: CO* | B: O*", loc='best',fontsize=6,title_fontsize=6)

            plt.show()

    #-------------------------------------------------------------------------------------------------------------------------------------
    def three_d_plot_iRates(self,p_c = 0.1):
        #iRates
        ##Constructing a waterfall plot
        from matplotlib.collections import PolyCollection
        from matplotlib.collections import LineCollection
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib import colors as mcolors
        import numpy as np
        import matplotlib

        New_out_df = self.Vis_Exp_MKM_Dataframe(p_c = p_c)

        if self.Input_Format=='KMC':
            exp_name ='KMC'

        Comp = 'iRates'
        
        g_sp = self.Exp_Gspecies #Gas Species list

        for i in np.arange(len(g_sp)):
            plt.figure(Comp+str(i))
            axes=plt.axes(projection="3d")

            def colors(arg):
                return mcolors.to_rgba(arg, alpha=6)

            x1 = New_out_df.loc[New_out_df['Sim_ndex'] == New_out_df.iloc[0][0]]['Time'].to_numpy()  #Sim_ndex = fist sim on the list and then find array of corresponding time
            verts1 = []
            verts2= []
            Sim_len = len(set(New_out_df['Sim_ndex']))
            sims = np.array(list(set(New_out_df['Sim_ndex'])))
            z1 = np.arange(Sim_len)
            for z in z1:
                df = New_out_df.loc[New_out_df['Sim_ndex'] == str(sims[z])] #Extracting only the dataframe corrsponding to simulation z
                KMC_vals = df[[col for col in df if exp_name+'_'+Comp+'_'+g_sp[i] in col]].to_numpy()
                MKM_vals = df[[col for col in df if 'MKM_'+Comp+'_'+g_sp[i] in col]].to_numpy()

                if g_sp[i]=='CO':
                    y1 = KMC_vals[:,0]
                    y2 = MKM_vals[:,0]
                else:
                    y1 = KMC_vals
                    y2 = MKM_vals

                verts1.append(list(zip(x1, y1)))
                verts2.append(list(zip(x1, y2)))

            facecolors = [matplotlib.cm.jet(x) for x in np.random.rand(Sim_len)]

            poly1 = LineCollection(verts1,color = facecolors,linewidths=(1,),zorder=2,linestyle='-')
            poly2 = LineCollection(verts2,color = facecolors,linewidths=(1,),zorder=2,linestyle='--')

            # Removes shaded region
            poly1.set_facecolor(None)
            poly2.set_facecolor(None)


            poly1.set_alpha(0.6)
            poly2.set_alpha(0.6)
            axes.add_collection3d(poly1, zs=z1, zdir='y')
            axes.add_collection3d(poly2, zs=z1, zdir='y')

            axes.set_xlabel('X : Time')
            axes.set_xlim3d(0, x1[-1])
            axes.set_ylabel('Y : Simulation')
            axes.set_ylim3d(0, Sim_len,auto=False)
            axes.set_yticklabels(np.array(list(set(New_out_df['Sim_names']))),fontdict={'fontsize': 6,'fontweight': 10})
            axes.set_zlabel(Comp)
            axes.set_zlim3d(-0.2, 0.2)
            axes.set_title(Comp+"_Results")

            line_1 = Line2D([0], [0], color='black', linewidth=0.6, linestyle='-') 
            line_2 = Line2D([0], [0], color='black', linewidth=0.6, linestyle='--')

            label_1 = exp_name+'_'+Comp+'_'+g_sp[i]
            label_2 = 'MKM_'+Comp+'_'+g_sp[i]

            lines = [line_1,line_2]
            labels = [label_1,label_2]
            axes.legend(lines, labels, title = "A: CO* | B: O*", loc='best',fontsize=6,title_fontsize=6)

            plt.show()


    #-------------------------------------------------------------------------------------------------------------------------------------
    def three_d_plot_Covg(self,p_c = 0.1):
        #Coverages
        ##Constructing a waterfall plot
        from matplotlib.collections import PolyCollection
        from matplotlib.collections import LineCollection
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib import colors as mcolors
        import numpy as np
        import matplotlib

        New_out_df = self.Vis_Exp_MKM_Dataframe(p_c = p_c)

        if self.Input_Format=='KMC':
            exp_name ='KMC'

        Comp = 'Covg'
        s_sp = self.Exp_Sspecies #Surface Species list
        g_sp = self.Exp_Gspecies #Gas Species list

        for i in np.arange(len(g_sp)):
            plt.figure(Comp+str(i))
            axes=plt.axes(projection="3d")

            def colors(arg):
                return mcolors.to_rgba(arg, alpha=6)

            x1 = New_out_df.loc[New_out_df['Sim_ndex'] == New_out_df.iloc[0][0]]['Time'].to_numpy()  #Sim_ndex = fist sim on the list and then find array of corresponding time
            verts1 = []
            verts2= []
            Sim_len = len(set(New_out_df['Sim_ndex']))
            sims = np.array(list(set(New_out_df['Sim_ndex'])))
            z1 = np.arange(Sim_len)
            for z in z1:
                df = New_out_df.loc[New_out_df['Sim_ndex'] == str(sims[z])] #Extracting only the dataframe corrsponding to simulation z
                KMC_vals = df[[col for col in df if exp_name+'_'+Comp+'_'+s_sp[i] in col]].to_numpy()
                MKM_vals = df[[col for col in df if 'MKM_'+Comp+'_'+s_sp[i] in col]].to_numpy()

                y1 = KMC_vals
                y2 = MKM_vals

                verts1.append(list(zip(x1, y1)))
                verts2.append(list(zip(x1, y2)))

            facecolors = [matplotlib.cm.jet(x) for x in np.random.rand(Sim_len)]

            poly1 = LineCollection(verts1,color = facecolors,linewidths=(1,),zorder=2,linestyle='-')
            poly2 = LineCollection(verts2,color = facecolors,linewidths=(1,),zorder=2,linestyle='--')

            # Removes shaded region
            poly1.set_facecolor(None)
            poly2.set_facecolor(None)


            poly1.set_alpha(0.6)
            poly2.set_alpha(0.6)
            axes.add_collection3d(poly1, zs=z1, zdir='y')
            axes.add_collection3d(poly2, zs=z1, zdir='y')

            axes.set_xlabel('X : Time')
            axes.set_xlim3d(0, x1[-1])
            axes.set_ylabel('Y : Simulation')
            axes.set_ylim3d(0, Sim_len,auto=False)
            axes.set_yticklabels(np.array(list(set(New_out_df['Sim_names']))),fontdict={'fontsize': 6,'fontweight': 10})
            axes.set_zlabel(Comp)
            axes.set_zlim3d(0, 1)
            axes.set_title(Comp+"_Results")

            line_1 = Line2D([0], [0], color='black', linewidth=0.6, linestyle='-') 
            line_2 = Line2D([0], [0], color='black', linewidth=0.6, linestyle='--')

            label_1 = exp_name+'_'+Comp+'_'+s_sp[i]
            label_2 = 'MKM_'+Comp+'_'+s_sp[i]

            lines = [line_1,line_2]
            labels = [label_1,label_2]
            axes.legend(lines, labels, title = "A: CO* | B: O*", loc='best',fontsize=6,title_fontsize=6)

            plt.show()

    #-------------------------------------------------------------------------------------------------------------------------------------
    # def MKM_Dataframe(self):
        
    #     if self.fit_k == False :
    #         exp_mkm_df = self.Exp_MKM_Dataframe(params = self.MKM.k)

    #     MKM_df = exp_mkm_df[exp_mkm_df.columns.drop(list(exp_mkm_df.filter(regex='KMC_')))]

    #     return MKM_df
    #-------------------------------------------------------------------------------------------------------------------------------------
    def Percent_diff_feat(self):
        rx,ry,rz = np.shape(self.EXP_Dictionary['iRates'])
        P_diff = np.zeros((rx,ry,rz)) 

        import math
        MKM_values = self.MKM_Dictionary['iRates']
        KMC_values = self.EXP_Dictionary['iRates']

        for i in np.arange(rx):
            for j in np.arange(ry):
                for k in np.arange(rz):
                    mkmr = KMC_values[i,j,k]
                    kmcr = MKM_values[i,j,k]      
                    #Preventing nan
                    if float(mkmr) == 0:
                        mkmr = 1e-20
                    if float(kmcr) == 0:
                        kmcr = 1e-20
                    val = abs(mkmr-kmcr)/((mkmr+kmcr)/2) 
                    P_diff[i,j,k] = val
                    if math.isinf(val) or math.isnan(val):
                        raise Exception('ERROR: inf or nan is present')
        return P_diff
        
    #-------------------------------------------------------------------------------------------------------------------------------------
    def log_ratio_corr_feat(self):
        rx,ry,rz = np.shape(self.EXP_Dictionary['iRates'])
        Corr_fac = np.zeros((rx,ry,rz)) 

        import math
        MKM_values = self.MKM_Dictionary['iRates']
        KMC_values = self.EXP_Dictionary['iRates']

        for i in np.arange(rx):
            for j in np.arange(ry):
                for k in np.arange(rz):
                    num = KMC_values[i,j,k]
                    den = MKM_values[i,j,k]
                    #Preventing log(0)
                    if float(num) == 0:
                        num = 1e-20
                    if float(den) == 0:
                        den = 1e-20
                        
                    frac = num/den
                    if float(frac) < 0: #(i.e the rates are either being calculated as consumed versus produced)
                        frac = abs(frac)
                        
                    val = np.log(frac)
                    Corr_fac[i,j,k] = val
                    if math.isinf(val) or math.isnan(val):
                        raise Exception('ERROR: inf or nan is present')
                    
        return Corr_fac

    #-------------------------------------------------------------------------------------------------------------------------------------
    def Full_Dataset_Dataframe(self):

        exp_mkm_df = self.Exp_MKM_Dataframe()

        if self.Input_Format=='KMC':
            #Adding Percent Diff
            gs_names = self.Exp_Gspecies
            for i in np.arange(self.KMC_n_gas_species):
                spec = gs_names[i]
                exp_mkm_df['P_diff_'+spec] = pd.DataFrame(self.Tensor_To_Array(self.Percent_diff_feat()))[1+i]

            #Adding Log Correc
            gs_names = self.Exp_Sspecies
            for i in np.arange(self.KMC_n_gas_species):
                spec = gs_names[i]
                exp_mkm_df['Corr_fac_'+spec] = pd.DataFrame(self.Tensor_To_Array(self.log_ratio_corr_feat()))[1+i]

        return exp_mkm_df
    
    #-------------------------------------------------------------------------------------------------------------------------------------
    def X_Y_full_dataset(self):
        
        out_df = self.Full_Dataset_Dataframe()
        out_df = out_df[out_df.columns.drop(list(out_df.filter(regex='KMC_')))] #Removing KMC Columns to leave it explicitly MKM

        All_columns = out_df.columns.to_list()
        target_columns = list(filter(lambda x: ('Corr') in x or ('P_diff') in x, All_columns))

        input_columns = [colmn for colmn in All_columns if colmn not in target_columns]

        X_all = out_df[input_columns] #Ignoring the first two columns(index and simulation name)
        Y_all = out_df[['Sim_ndex','Sim_names']+target_columns]

        return X_all,Y_all
    
    #-------------------------------------------------------------------------------------------------------------------------------------
    def X_Y_train_test_split(self,p_c = None):

        if p_c==None:
            p_c = self.test_train_split
        else:
            p_c = p_c
            

        import random
        print('Percent of test data selected:',p_c*100,'%')

        X_all,Y_all = self.X_Y_full_dataset()

        out_df = self.Full_Dataset_Dataframe()

        All_columns = out_df.columns.to_list()
        target_columns = list(filter(lambda x: ('Corr') in x or ('P_diff') in x, All_columns))

        input_columns = [colmn for colmn in All_columns if colmn not in target_columns]
        
        max_sim_number = int(X_all[input_columns[0]].iloc[-1]) #MAx_number of simulations present #Count is starting from 0
        n_test_sim = int(p_c*max_sim_number) #Number of simulations being used as test
        sim_nums = list(set(X_all['Sim_ndex'])) # List of unique simulation numbers

        test_sims = random.sample(sim_nums,n_test_sim) #Random sim_numbers for testing
        print('\n','The list of simulations used in the test dataset:\n',test_sims)

        X_test = X_all.loc[X_all['Sim_ndex'].isin(test_sims)]
        Y_test = Y_all.loc[Y_all['Sim_ndex'].isin(test_sims)]

        X_train = X_all[~X_all['Sim_ndex'].isin(test_sims)]
        Y_train = Y_all[~Y_all['Sim_ndex'].isin(test_sims)]

        #REMOVING THE SIM_NDEX AND SIM_NAMES COLUMNS
        X_test = X_test.drop(columns=['Sim_ndex','Sim_names'])
        Y_test = Y_test.drop(columns=['Sim_ndex','Sim_names'])

        X_train = X_train.drop(columns=['Sim_ndex','Sim_names'])
        Y_train = Y_train.drop(columns=['Sim_ndex','Sim_names'])

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        return X_train,Y_train,X_test,Y_test
    #-------------------------------------------------------------------------------------------------------------------------------------
    def ML_model(self,X_train, Y_train, algorithm=None):

        if algorithm!=None:
            self.ML_algorithm = algorithm
        else:
            algorithm = self.ML_algorithm

        #XGBoost Algorithm
        #https://xgboost.readthedocs.io/en/stable/python/python_api.html
        if algorithm=="XGBoost":  
            import xgboost as xgb

            reg = xgb.XGBRegressor(booster='gbtree',    
                                n_estimators=1500,
                                objective='reg:squarederror',
                                max_depth=20,
                                learning_rate=0.01)
            reg.fit(X_train, Y_train,
                    eval_set=[(X_train, Y_train)],
                    verbose=False)
        
        #Artificial Neural Network
        #https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
        elif algorithm=="ANN":
            from sklearn.neural_network import MLPRegressor
            No_H_nodes_per_layer = 128
            print('Number of Hidden layer nodes per layer : ',No_H_nodes_per_layer)
            No_H_layers = 4
            print('Number of Hidden layers: ',No_H_layers)

            hidden_layers = No_H_nodes_per_layer*np.ones(No_H_layers) 
            hidden_layer_sizes = tuple(tuple(int(item) for item in hidden_layers))
            reg = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                activation='relu', 
                                solver='adam')
    #                            ,random_state=42, 
    #                             max_iter=300)

            reg.fit(X_train, Y_train)
        
        #K-Nearest Neighbor
        #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
        elif algorithm=="KNN":
            from sklearn.neighbors import KNeighborsRegressor

            reg = KNeighborsRegressor(n_neighbors=50, weights='distance',p=1)
            reg.fit(X_train, Y_train)
        
        #RandomForest 
        #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
        elif algorithm=='RandomForest':
            from sklearn.ensemble import RandomForestRegressor
            
            reg = RandomForestRegressor(n_estimators=1000, random_state=42)

            reg.fit(X_train, Y_train)
            
            print('Features:',X_train.columns)
            
            print('\nFeature Importance:\n',reg.feature_importances_) #Shows which features are chosen most when doing splits #gives the most information
            
        elif algorithm=='DecisionTree':
            from sklearn import tree
            reg = tree.DecisionTreeRegressor()#criterion='poisson',max_depth=20,min_samples_leaf=10,min_samples_split=20
            
            reg.fit(X_train, Y_train)
            
            print('Features:',X_train.columns)
            
            print('\nFeature Importance:\n',reg.feature_importances_) #Shows which features are chosen most when doing splits #gives the most information

        return reg

    #-------------------------------------------------------------------------------------------------------------------------------------
    def ML_Model_Select(self, ALGORITHM_NAME = None):
        if ALGORITHM_NAME!=None:
            self.ML_algorithm = ALGORITHM_NAME
        else:
            ALGORITHM_NAME = self.ML_algorithm


        import time
        ######### OPTIONS: 'XGBoost','ANN','KNN','RandomForest'#########
        ################################################################
        # ALGORITHM_NAME = "KNN"
        ################################################################

        X_train,Y_train,X_test,Y_test = self.X_Y_train_test_split()

        start_time = time.time()
        reg = self.ML_model(X_train, Y_train)
        end_time = time.time()

        elapsed_time = end_time - start_time
        print("\nElapsed Model Training Time: \n", elapsed_time, "seconds \n", elapsed_time/60, "minutes")

        return reg
    #-------------------------------------------------------------------------------------------------------------------------------------
    def External_Dataframe(self):

        if self.Input_Format=='KMC':
                
            KMC_Data_EXP = self.Input
            self.Data_EXP_rates= KMC_Data_EXP.iloc[:,-self.KMC_n_gas_species:] #To be used to later to compare and asses ML prediction results

            #Creating repeated init covs matrix
            test_data_ini_cov = KMC_Data_EXP.iloc[0,1:5].values
            test_data_time_interv = KMC_Data_EXP.iloc[:,0].values
            matrix_test_data_ini_cov = np.empty((len(test_data_time_interv),len(test_data_ini_cov)))
            for i in np.arange(len(test_data_time_interv)):
                matrix_test_data_ini_cov[i] = test_data_ini_cov
        
            self.MKM_test_inp.set_limits_of_integration(Ti=float(KMC_Data_EXP['Time'].head(1)),Tf=float(KMC_Data_EXP['Time'].tail(1)))
            self.MKM_test_inp.ODE_Tolerances(Dplace=50,reltol=1e-8,abstol=1e-8)

            self.MKM_test_inp.k = self.forced_param #From fitting or external

            MKM_Covg_test_inp = np.zeros((len(test_data_time_interv),len(test_data_ini_cov))) # O*, CO*, O2*, *
            MKM_Rates_test_inp = np.zeros((len(test_data_time_interv),self.KMC_n_gas_species)) # O2, CO, CO2

            self.MKM_test_inp.set_initial_coverages(init=test_data_ini_cov)

            sola,solta = self.MKM_test_inp.solve_coverage(Tf_eval=test_data_time_interv,plot=False)
            MKM_Covg_test_inp = sola #Coverage profile matrix

            solb,soltb = self.MKM_test_inp.solve_rate_production(Tf_eval=test_data_time_interv,plot=False)
            MKM_Rates_test_inp = (solb[:,0:self.KMC_n_gas_species])

            self.Exp_surf_name = KMC_Data_EXP.columns.to_list()[1:self.KMC_n_surf_species+1]
            self.Exp_gas_name = [i[2:] for i in KMC_Data_EXP.columns.to_list()[self.KMC_n_surf_species+1:]]

            Test_input = pd.DataFrame()

            #Adding initial coverages
            surf_names = self.Exp_surf_name
            for i in np.arange(self.KMC_n_surf_species):
                spec = surf_names[i]
                Test_input['Init_Covg_'+spec] = pd.DataFrame(matrix_test_data_ini_cov)[i]

            #Adding Time
            Test_input['Time'] = pd.DataFrame(test_data_time_interv)

            #Adding coverage profiles of surface species
            surf_names = self.Exp_surf_name
            for i in np.arange(self.KMC_n_surf_species):
                spec = surf_names[i]
                Test_input['MKM_Covg_'+spec] = pd.DataFrame(MKM_Covg_test_inp)[i]
                
            #Adding iRates profiles of gaseous species
            gs_names = self.Exp_gas_name
            for i in np.arange(self.KMC_n_gas_species):
                spec = gs_names[i]
                Test_input['MKM_iRates_'+spec] = pd.DataFrame(MKM_Rates_test_inp)[i]


            return Test_input
        
    #-------------------------------------------------------------------------------------------------------------------------------------
    def ML_Fitting(self, alg = None, test_train_split=None, plot=True):
        if test_train_split != None:
            self.test_train_split = test_train_split            
        
        if alg!=None:
            self.ML_algorithm = alg
        else:
            alg = self.ML_algorithm

        reg = self.ML_Model_Select()

        Test_input  = self.External_Dataframe()
        self.external_data_input = Test_input #For program outputing purposes
        
        Test_output = reg.predict(Test_input)
        self.external_data_output = Test_output #For program outputing purposes
        
        self.Predicted_ML_output = Test_output

        if self.Input_Format == 'KMC':

            test_data_time_interv = np.array(Test_input['Time'])

            Pred_corr = Test_output[:,-len(self.Exp_Gspecies):] #extracting correction factors  #O2 #CO #CO2
            
            KMC_Data_EXP_rates = self.Data_EXP_rates #KMC_INPUT_RATE_DATA
            
            MKM_Rates_test_inp = Test_input[[col for col in Test_input if 'MKM_'+'iRates'+'_' in col]].to_numpy()
            #Calculating ML corrected predicted rates
            ML_Rates_pred = np.zeros((len(test_data_time_interv),len(self.Exp_Gspecies)))  #O2, #CO, CO2
            for i in np.arange(np.shape(ML_Rates_pred)[0]):
                for j in np.arange(np.shape(ML_Rates_pred)[1]):
                    ML_Rates_pred[i,j] = MKM_Rates_test_inp[i,j]*np.exp(Pred_corr[i,j])
                    
            from math import sqrt
            from sklearn.metrics import mean_squared_error
            #Calculating the root mean squared of the test set
            print('Root Mean Squared Error:\n',sqrt(mean_squared_error(KMC_Data_EXP_rates, ML_Rates_pred)))

            if plot==True:

                plt.figure(figsize = (8, 6))
                plt.plot(test_data_time_interv, KMC_Data_EXP_rates.values[:,0],'r*', label='O2_kMC')        
                plt.plot(test_data_time_interv, KMC_Data_EXP_rates.values[:,1],'g*', label='CO_kMC') 
                plt.plot(test_data_time_interv, KMC_Data_EXP_rates.values[:,2], 'b*', label='CO2_kMC') 

                plt.plot(test_data_time_interv, MKM_Rates_test_inp[:,0],'ro', label='O2_MKM')        
                plt.plot(test_data_time_interv, MKM_Rates_test_inp[:,1],'go', label='CO_MKM') 
                plt.plot(test_data_time_interv, MKM_Rates_test_inp[:,2], 'bo', label='CO2_MKM') 

                plt.plot(test_data_time_interv, ML_Rates_pred[:,0],'r-', label='O2_ML')        
                plt.plot(test_data_time_interv, ML_Rates_pred[:,1],'g-', label='CO_ML') 
                plt.plot(test_data_time_interv, ML_Rates_pred[:,2], 'b-', label='CO2_ML') 

                plt.xlabel('Time, s')
                plt.ylabel("Rates_production, $r$")
                plt.title('ML_rate_correction_Results')
                # plt.ylim([-0.2,0.2])
                plt.legend(fontsize=5, loc='best')
                plt.show()

                return plt.figure







