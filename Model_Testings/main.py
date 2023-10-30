from scipy.integrate import solve_ivp   #ODE solver
import matplotlib.pyplot as plt         #package for plotting
import numpy as np   #package for numerical arithmetics and analysis
import pandas as pd  #package for dataframe and file extraction/creation
import string        #package to allow for access to alphabet strings
from mpmath import * #package for precision control
dplace=10    #Controls decimal places - used for mp.dps in mpmath precision control
from scipy import optimize
import sys, os
from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing

# Disable
def blockPrint():
    sys.__stdout__ = sys.stdout
    sys.stdout = open(os.devnull, 'w')

# Restore
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
        if (np.sum(vec))!=1 or (all(x >= 0 for x in vec)!=True) or (all(x <= 1 for x in vec)!=True):
            raise Exception('Error: The initial coverages entered are not valid. \n Please double check the initial coverages entered and make the necessary corrections')
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
    def set_initial_coverages(self,init=[]): #empty sites included at the end of the code 
        mp.dps= dplace
        
        ExpNoCovg = len(self.Stoich.iloc[0,len(self.P)+1:])
        if init==[]: 
            init=np.zeros(ExpNoCovg-1)
            
        if len(init)!=(ExpNoCovg-1):
            raise Exception('Number of coverage entries do not match what is required. %i entries are needed. (Not including number/coverage of empty sites).'%(ExpNoCovg-1))
        else: 
            #Changing the number of decimal places/precision of input coverages
            for i in np.arange(len(init)):
                init[i]=mpf(init[i])
                
            empty_sites = 1 - np.sum(init)
            self.init_cov = np.append(init,empty_sites)
                
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
    def get_rates(self,cov=[]): #u = coverages (excluding empty sites) #Function used to calculate the rates of reactions
        
        if cov==[]:
            cov=self.init_cov
            
        THETA = cov #Coverages being investigated
        
        
        Nr = len(self.Stoich) #Number of rows in your your stoich matrix, i.e (Number of reactions)
       

        kf = self.k[0::2] #Pulling out the forward rxn rate constants (::2 means every other value, skip by a step of 2)
        kr = self.k[1::2] #Pulling out the reverse rxn rate constants 
        

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
                    
            r[j] = (kf[j]*np.prod(fwd)) - (kr[j]*np.prod(rvs)) #Calculating the rate of reaction
         
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
    def solve_coverage(self,t=[],initial_cov=[],method='BDF',reltol=1e-6,abstol=1e-8,Tf_eval=None,full_output=False,plot=False): #Function used for calculating (and plotting) single state transient coverages
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
        if Tf_eval==None:
            T_eval=None
        else:
            T_eval=np.linspace(0, Tf_eval, num=1000)
        
        solve = solve_ivp(self.get_ODEs,t_span,init,method,t_eval=T_eval,rtol=reltol,atol=abstol,dense_output=full_output) #ODE Solver
        
        #COnvergence Check
        if solve.status!=0:
            self.status = 'Convergence Failed'
            raise Exception('ODE Solver did not successfuly converge. Please check model or tolerances used')
        elif solve.status==0:
            self.status = 'ODE Solver Converged'
            print(self.status)
        
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
    def solve_rate_reaction(self,tf=None,initial_coverage=[],plot=False): #Function used for calculating (and plotting) single state transient rates of reaction
        
        if tf==None: 
            tf=self.Tf
        covg,covgt =self.solve_coverage(t=[self.Ti,tf],initial_cov=initial_coverage)
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
    def solve_rate_production(self,tf=None,initial_coverage=[],plot=False): #Function used for calculating (and plotting) single state transient rates of production
        
        if tf==None:
            tf=self.Tf
        covg,covgt =self.solve_coverage(t=[self.Ti,tf],initial_cov=initial_coverage)
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
    def get_SS_coverages(self,tf=None): #Function used for calculating the steady state coverages
        if tf==None:
            tf=self.Tf
            
        covg,covgt = self.solve_coverage(t=[self.Ti,tf])
        
        SS,msg = self.check_SS(covg,feature='coverage')
        print(msg)
        return SS
    #------------------------------------------------------------------------------------------------------------------------------    
    def get_SS_rates_reaction(self,tf=None): #Function used for calculating the steady state rates of reaction
        rates_r,time_r = self.solve_rate_reaction(tf=tf)
        
        SS,msg = self.check_SS(rates_r,feature='rates_reaction')
        print(msg)
        return SS
    #------------------------------------------------------------------------------------------------------------------------------
    def get_SS_rates_production(self,tf=None): #Function used for calculating the steady state rates of production
        rates_p,time_R = self.solve_rate_production(tf=tf)  
        
        SS,msg = self.check_SS(rates_p,feature='rates_production')
        print(msg)
        return SS
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
            SS_State1 = self.get_SS_coverages()[:-1]
            enablePrint() #Re-enable printing
        else:
            self.set_rxnconditions(Pr=State1)
            SS_State1 = self.get_SS_coverages()[:-1]

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
    def create_csv(self,sol,solt,Name=None,label=None):
        
        if label==None:
            label=self.label  #Using the most recent label
        
        if label not in ('coverages','rates_r','rates_p'): #Making sure one of the labels is chosen (i.e not None)
            raise Exception("The entered label is incorrect. Please insert either 'coverage' or 'rates_r' or 'rates_p' ")
        
        if Name!=None and Name[-4:]!='.csv':  #Making sure that the Name entered has .csv attachment
            raise Exception("Name entered must end with .csv ; Example coverages.csv")
            
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
            ax.set_ylabel(r"Rates of Production, $R_i, [TOF]$")
            ax.set_title('Rates of production versus Time')
            
        elif label=='rates_r':
            ax.legend(np.array(self.Stoich.iloc[:,0]),fontsize=10, loc='upper right',facecolor='white', edgecolor ='black', framealpha=1)
            ax.set_xlabel('Time, t, [s]')
            ax.set_ylabel(r"Rates of Reaction, $r_i, [TOF]$")
            ax.set_title('Rates of reaction versus Time')
            
        elif label=='coverages':
            ax.legend(self.Atomic.columns.values[1+len(self.P):],fontsize=10, loc='upper right',facecolor='white', edgecolor ='black', framealpha=1)
            ax.set_xlabel('Time, t, [s]')
            ax.set_ylabel(r"Coverage, $\theta_i, [ML]$")
            ax.set_title('Coverages versus Time')
#------------------------------------------------------------------------------------------------------------------------------            
class MKModel_wCD:
    def __init__(self,Atomic_csv,Stoich_csv,Param_csv): #Inputs necessary to initialize the MK Model
        self.Atomic = pd.read_csv(Atomic_csv)     #Opening/Reading the Atomic input file needed to be read
        self.Stoich = pd.read_csv(Stoich_csv)    #Opening/Reading the Stoichiometric input file needed to be read
        self.Param = pd.read_csv(Param_csv)     #Opening/Reading the Parameter input file needed to be read
        
        self.MKM = MKModel(Atomic_csv,Stoich_csv,Param_csv)
                
        self.k = self.kextract()    #Extracting the rate constants from the Param File (Note that format of the Param File is crucial)
        self.P ,self.Temp = self.set_rxnconditions() #Setting reaction conditions (defaulted to values from the Param File but can also be set mannually )
        self.Coeff = self.Coeff_extract() #Extracting the coverage dependance coefficients
        self.Ti,self.Tf=self.set_limits_of_integration() #Sets the range of time needed to solve for the relavant MK ODEs, defaults to 0-6e6 but can also be manually set
        self.init_cov=self.set_initial_coverages() #Sets the initial coverage of the surface species, defaults to zero coverage but can also be set manually
        self.status='Waiting' #Used to observe the status of the ODE Convergence
        self.label='None'   #Used to pass in a label so as to know what kind of figure to plot
    #------------------------------------------------------------------------------------------------------------------------------    
    def check_coverages(self,vec):  #Function to check if the coverages being inputted make sense (Note in this code empty sites are not inputted, they're calculated automatically)
        return self.MKM.check_coverages(vec)
    #------------------------------------------------------------------------------------------------------------------------------   
    def check_massbalance(self,Atomic,Stoich): #Function to check if mass is balanced
        return self.MKM.check_massbalance(self.Atomic,self.Stoich)
    #------------------------------------------------------------------------------------------------------------------------------    
    def Pextract(self): #Function used for extracting pressures from the Param File
        return self.MKM.Pextract()
    #------------------------------------------------------------------------------------------------------------------------------
    def kextract(self): #Function used for extracting rate constants from the Param File
        return self.MKM.kextract()
    #------------------------------------------------------------------------------------------------------------------------------
    def set_initial_coverages(self,init=[]): #empty sites included at the end of the code
        mp.dps= dplace
        
        ExpNoCovg = len(self.Stoich.iloc[0,len(self.P)+1:])
        if init==[]: 
            init=np.zeros(ExpNoCovg-1)
            
        if len(init)!=(ExpNoCovg-1):
            raise Exception('Number of coverage entries do not match what is required. %i entries are needed. (Not including number/coverage of empty sites).'%(ExpNoCovg-1))
        else: 
            #Changing the number of decimal places/precision of input coverages
            for i in np.arange(len(init)):
                init[i]=mpf(init[i])
                
            empty_sites = 1 - np.sum(init)
            self.init_cov = np.append(init,empty_sites)
                
        return self.check_coverages(self.init_cov)
    #------------------------------------------------------------------------------------------------------------------------------    
    def set_rxnconditions(self,Pr=None,Temp=None): #Function used for setting the reaction Pressure and Temperature (Currently Temperature is not in use)
        self.P,self.Temp = self.MKM.set_rxnconditions(Pr,Temp)
        return self.P,self.Temp
    #------------------------------------------------------------------------------------------------------------------------------
    def set_limits_of_integration(self,Ti=0,Tf=6e6): #Function used for setting the time limits of integration 
        self.Ti,self.Tf=self.MKM.set_limits_of_integration(Ti,Tf)
        return self.Ti,self.Tf
    #------------------------------------------------------------------------------------------------------------------------------
    def Coeff_extract(self):
        colmn = len(self.Stoich.iloc[0,1:]) - len(self.P) - 1 #Number of columns (i.e rate coefficients = no. of surface species being investigated)
        row = len(self.k) #number of rows = number of rate constants (i.e reaction steps)
        Coeff = np.empty([row,colmn]) #initializing the coefficient matrix
        index = list(string.ascii_lowercase)[:colmn] #index holding the relevant letters (a-w) (Therefor 23 different species only possible)
        
        #Extracting the coefficients from the Param matrix
        for i in np.arange(colmn):
            count = 0
            for j in np.arange(len(self.Param.iloc[:,0])):
                if ('const' == self.Param.iloc[j,0]) and (str(index[i]) in self.Param.iloc[j,1]):
                    Coeff[count][i]=self.Param.iloc[j,2]
                    count += 1     
        return Coeff
    #------------------------------------------------------------------------------------------------------------------------------
    def ratecoeff(self,kref,Coeff,Theta):
        if len(Coeff) != len(Theta):
            raise Exception('The number of the coefficients doesnt match the relevant coverages. Please make sure to check the Parameters csv file for any errors. ')
        else:
            K = kref*np.exp(float(np.sum(np.multiply(Coeff,Theta))))  #/RT lumped into a and b assuming T is constant
            return K
    #------------------------------------------------------------------------------------------------------------------------------
    def get_rates(self,cov=[]): #u = coverages (excluding empty sites) #Function used to calculate the rates of reactions
        
        if cov==[]:
            cov=self.init_cov
            
        THETA = cov #Coverages being investigated

        Nr = len(self.Stoich) #Number of rows in your your stoich matrix, i.e (Number of reactions)
       

        kf = self.k[0::2] #Pulling out the forward rxn rate constants (::2 means every other value, skip by a step of 2)
        kr = self.k[1::2] #Pulling out the reverse rxn rate constants 
        
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
                    
            r[j] = (self.ratecoeff(kf[j],Coeff_f[j][:],THETA[:-1])*np.prod(fwd)) - (self.ratecoeff(kr[j],Coeff_r[j][:],THETA[:-1])*np.prod(rvs)) #Calculating the rate of reaction
         
        r = np.transpose(r)
        
        return r    
    #------------------------------------------------------------------------------------------------------------------------------    
    def get_ODEs(self,t,cov,coverage=True): #t only placed for solve_ivp purposes #Functions used for calculating the rates of production
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
    def solve_coverage(self,t=[],initial_cov=[],method='BDF',reltol=1e-6,abstol=1e-8,Tf_eval=None,full_output=False,plot=False): #Function used for calculating (and plotting) single state transient coverages
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
        if Tf_eval==None:
            T_eval=None
        else:
            T_eval=np.linspace(0, Tf_eval, num=1000)
        
        solve = solve_ivp(self.get_ODEs,t_span,init,method,t_eval=T_eval,rtol=reltol,atol=abstol,dense_output=full_output) #ODE Solver
        
        #Convergence Check
        if solve.status!=0:
            self.status = 'Convergence Failed'
            raise Exception('ODE Solver did not successfuly converge. Please check model or tolerances used')
        elif solve.status==0:
            self.status = 'ODE Solver Converged'
            print(self.status)
        
        #Extracting the Solutions:
        sol = np.transpose(solve.y)
        solt = np.transpose(solve.t)
        
        self.label='coverages'
        # Plotting
        
        if plot==False:
            return sol,solt
        elif plot==True:
            self.MKM.plotting(sol,solt,self.label)
            return sol,solt
    #------------------------------------------------------------------------------------------------------------------------------
    def solve_rate_reaction(self,tf=None,initial_coverage=[],plot=False): #Function used for calculating (and plotting) single state transient rates of reaction
        
        if tf==None: 
            tf=self.Tf
        covg,covgt =self.solve_coverage(t=[self.Ti,tf],initial_cov=initial_coverage)
        rates_r = []
        for t in np.arange(len(covgt)):
            rates_r.append(self.get_rates(cov = covg[t,:]))
                        
        rates_r = np.array(rates_r)
        
        self.label='rates_r'
        if plot==False:
            return rates_r,covgt
        elif plot==True:
            self.MKM.plotting(rates_r,covgt,self.label)
            return rates_r,covgt
    #------------------------------------------------------------------------------------------------------------------------------
    def solve_rate_production(self,tf=None,initial_coverage=[],plot=False): #Function used for calculating (and plotting) single state transient rates of production
        
        if tf==None:
            tf=self.Tf
        covg,covgt =self.solve_coverage(t=[self.Ti,tf],initial_cov=initial_coverage)
        rates_p = []
        for t in np.arange(len(covgt)):
            rates_p.append(self.get_ODEs(covgt[t],covg[t,:],coverage=False))
                        
        rates_p = np.array(rates_p)
        
        self.label='rates_p'
        if plot==False:
            return rates_p,covgt
        elif plot==True:
            self.MKM.plotting(rates_p,covgt,self.label)
            return rates_p,covgt
    #------------------------------------------------------------------------------------------------------------------------------ 
    #Functions neccessary for calculating the steady state values (Needed for when attempting dynamic switching between pressures)
    #------------------------------------------------------------------------------------------------------------------------------    
    def check_SS(self,trans_vec,tol=0.10,feature=None): #Function for checking if steady state has been reached
        return self.MKM.check_SS(trans_vec=trans_vec,tol=tol,feature=feature)
    #------------------------------------------------------------------------------------------------------------------------------
    def get_SS_coverages(self,tf=None): #Function used for calculating the steady state coverages
        if tf==None:
            tf=self.Tf
            
        covg,covgt = self.solve_coverage(t=[self.Ti,tf])
        
        SS,msg = self.check_SS(covg,feature='coverage')
        print(msg)
        return SS
    #------------------------------------------------------------------------------------------------------------------------------
    def get_SS_rates_reaction(self,tf=None): #Function used for calculating the steady state rates of reaction
        rates_r,time_r = self.solve_rate_reaction(tf=tf)
        
        SS,msg = self.check_SS(rates_r,feature='rates_reaction')
        print(msg)
        return SS
    #------------------------------------------------------------------------------------------------------------------------------
    def get_SS_rates_production(self,tf=None): #Function used for calculating the steady state rates of production
        rates_p,time_R = self.solve_rate_production(tf=tf)  
        
        SS,msg = self.check_SS(rates_p,feature='rates_production')
        print(msg)
        return SS
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
            SS_State1 = self.get_SS_coverages()[:-1]
            enablePrint() #Re-enable printing
        else:
            self.set_rxnconditions(Pr=State1)
            SS_State1 = self.get_SS_coverages()[:-1]

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
            self.MKM.plotting(sol,solt,self.label)
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
            self.MKM.plotting(sol,solt,self.label)
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
            self.MKM.plotting(sol,solt,self.label)
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
                self.MKM.plotting(full_covg,full_time,self.label)
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
                self.MKM.plotting(full_rt_r,full_time,self.label)
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
                self.MKM.plotting(full_rt_p,full_time,self.label)
                return full_rt_p,full_time
    #------------------------------------------------------------------------------------------------------------------------------    
    def create_csv(self,sol,solt,Name=None,label=None):
        self.label = label
        return self.MKM.create_csv(sol,solt,Name=Name,label=label)
#------------------------------------------------------------------------------------------------------------------------------    
class Fitting:    
    def __init__(self,Input_csv,Atomic_csv,Stoich_csv,Param_Guess_csv,CovgDep=False): #Inputs necessary to initialize the MK Model
        self.Input = pd.read_csv(Input_csv)
        self.Atomic = pd.read_csv(Atomic_csv)     #Opening/Reading the Atomic input file needed to be read
        self.Stoich = pd.read_csv(Stoich_csv)    #Opening/Reading the Stoichiometric input file needed to be read
        self.Param_Guess = pd.read_csv(Param_Guess_csv)     #Opening/Reading the Parameter of guess input file needed to be read
        self.CovgDep = CovgDep
        
        if self.CovgDep==False:
            self.MKM = MKModel(Atomic_csv,Stoich_csv,Param_Guess_csv)
        elif self.CovgDep==True:    
            self.MKM = MKModel_wCD(Atomic_csv,Stoich_csv,Param_Guess_csv)
            self.Coeff = self.Coeff_extract() #Extracting the coverage dependance coefficients
                
        self.k = self.kextract()    #Extracting the rate constants from the Param File (Note that format of the Param File is crucial)
        self.P ,self.Temp = self.set_rxnconditions() #Setting reaction conditions (defaulted to values from the Param File but can also be set mannually )
        self.Ti,self.Tf=self.set_limits_of_integration() #Sets the range of time needed to solve for the relavant MK ODEs, defaults to 0-6e6 but can also be manually set
        self.init_cov=self.set_initial_coverages() #Sets the initial coverage of the surface species, defaults to zero coverage but can also be set manually
        self.status='Waiting' #Used to observe the status of the ODE Convergence
        self.label='None'   #Used to pass in a label so as to know what kind of figure to plot
    #------------------------------------------------------------------------------------------------------------------------------    
    def check_coverages(self,vec):  #Function to check if the coverages being inputted make sense (Note in this code empty sites are not inputted, they're calculated automatically)
        return self.MKM.check_coverages(vec)
    #------------------------------------------------------------------------------------------------------------------------------   
    def check_massbalance(self,Atomic,Stoich): #Function to check if mass is balanced
        return self.MKM.check_massbalance(self.Atomic,self.Stoich)
    #------------------------------------------------------------------------------------------------------------------------------    
    def Pextract(self): #Function used for extracting pressures from the Param File
        return self.MKM.Pextract()
    #------------------------------------------------------------------------------------------------------------------------------
    def kextract(self): #Function used for extracting rate constants from the Param File
        return self.MKM.kextract()
    #------------------------------------------------------------------------------------------------------------------------------    
    def set_rxnconditions(self,Pr=None,Temp=None): #Function used for setting the reaction Pressure and Temperature (Currently Temperature is not in use)
        self.P,self.Temp = self.MKM.set_rxnconditions(Pr,Temp)
        return self.P,self.Temp
    #------------------------------------------------------------------------------------------------------------------------------
    def set_initial_coverages(self,init=[]): #empty sites included at the end of the code
        return self.MKM.set_initial_coverages()
    #------------------------------------------------------------------------------------------------------------------------------
    def set_limits_of_integration(self,Ti=0,Tf=6e6): #Function used for setting the time limits of integration 
        self.Ti,self.Tf=self.MKM.set_limits_of_integration(Ti,Tf)
        return self.Ti,self.Tf
    #------------------------------------------------------------------------------------------------------------------------------
    def Coeff_extract(self):
        return self.MKM.Coeff_extract()
    #------------------------------------------------------------------------------------------------------------------------------    
    def extract(self):
        n=30
        lnt = len(self.Input.iloc[:,0])
        inp_array = self.Input.to_numpy()
        dist = len(inp_array[:,0][::round(lnt/n)])
        
        Ext_inp = np.empty((dist,len(self.Input.iloc[0,:]))) #Extracted n values from input

        for i in np.arange(len(self.Input.iloc[0,:])):
            Ext_inp[:,i]=inp_array[:,i][::round(lnt/n)]
            
        return Ext_inp
    #------------------------------------------------------------------------------------------------------------------------------    
    def normalize(self,Ext_inp=[]):
        if Ext_inp==[]:
            Ext_inp=self.extract()
        
        Norm_inp = np.empty(np.shape(Ext_inp))
        for i in np.arange(len(Ext_inp[0,:])):
            mi = min(Ext_inp[:,i])
            ma = max(Ext_inp[:,i])
            Norm_inp[:,i]=(Ext_inp[:,i]-mi)/(ma-mi)
        
        return Norm_inp
    #------------------------------------------------------------------------------------------------------------------------------      
    def solve_coverage(self,t=[],initial_cov=[],method='BDF',reltol=1e-6,abstol=1e-8,Tf_eval=[],full_output=False,plot=False): #Function used for calculating (and plotting) single state transient coverages
        #Function used for solving the resulting ODEs and obtaining the corresponding surface coverages as a function of time
        #re-written for T_eval capabilities
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
        
        solve = solve_ivp(self.MKM.get_ODEs,t_span,init,method,t_eval=T_eval,rtol=reltol,atol=abstol,dense_output=full_output) #ODE Solver
        
        #COnvergence Check
        if solve.status!=0:
            self.status = 'Convergence Failed'
            raise Exception('ODE Solver did not successfuly converge. Please check model or tolerances used')
        elif solve.status==0:
            self.status = 'ODE Solver Converged'
            print(self.status)
        
        #Extracting the Solutions:
        sol = np.transpose(solve.y)
        solt = np.transpose(solve.t)
        
        self.label='coverages'
        
        # Plotting
        
        if plot==False:
            return sol,solt
        elif plot==True:
            self.MKM.plotting(sol,solt,self.label)
            return sol,solt
    #------------------------------------------------------------------------------------------------------------------------------    
    # Cost/Minimization Functions
    #------------------------------------------------------------------------------------------------------------------------------    
    def covg_func(self,x,*fit_params):
        self.fit_params = fit_params
        colmn = len(self.Stoich.iloc[0,1:]) - len(self.P) - 1 #Number of columns (i.e rate coefficients = no. of surface species being investigated)
        rw = len(self.k)
        
        if self.CovgDep==False:
            self.MKM.k = self.fit_params
        elif self.CovgDep==True:
            self.MKM.k = self.fit_params[:rw]
            self.MKM.Coeff = np.reshape(self.fit_params[rw:],(rw,colmn))
        
        input_time=self.extract()[:,0]
        inp_init_covg = self.extract()[0,1:-1]
        sol,solt= self.solve_coverage(t=[0,input_time[-1]],initial_cov=inp_init_covg,Tf_eval=input_time) #Uses MKM.getODEs, but the inclass solve_coverage to add custom time dependancies
        dat = np.insert(sol,0,solt,axis=1)   #Merging time and parameters
        Norm_sol = self.normalize(Ext_inp=dat)

        return np.reshape(Norm_sol[:,1:],Norm_sol[:,1:].size)
    #------------------------------------------------------------------------------------------------------------------------------    
    def cost_func_minimize(self,fit_params):
        self.fit_params = fit_params
        colmn = len(self.Stoich.iloc[0,1:]) - len(self.P) - 1 #Number of columns (i.e rate coefficients = no. of surface species being investigated)
        rw = len(self.k)
        og = self.normalize() #Original input
        
        if self.CovgDep==False:
            self.MKM.k = self.fit_params
        elif self.CovgDep==True:
            self.MKM.k = self.fit_params[:rw]
            self.MKM.Coeff = np.reshape(self.fit_params[rw:],(rw,colmn))
        
        # print(self.MKM.k)    
        input_time=self.extract()[:,0]
        inp_init_covg = self.extract()[0,1:-1]
        sol,solt= self.solve_coverage(t=[0,input_time[-1]],initial_cov=inp_init_covg,Tf_eval=input_time) #Uses MKM.getODEs, but the inclass solve_coverage to add custom time dependancies
        dat = np.insert(sol,0,solt,axis=1)   #Merging time and parameters
        Norm_sol = self.normalize(Ext_inp=dat)
        
        ns = len(Norm_sol[0,1:]) #Number of species
        w = self.min_weight*np.ones(ns)
        error_t=[]
        for i in np.arange(ns):
            error_t.append(w[i]*(og[:,(i+1)] - Norm_sol[:,i])**2)
            
        error_t = np.sum(error_t,axis=1)    
        error = (np.sum(error_t))
        print(error)
        print(fit_params)
        return error
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
    def curve_fit_func(self,method,maxfev,xtol,ftol):
        values = self.normalize()

        x_values = values[:,0]
        y_values = np.reshape(values[:,1:],values[:,1:].size)
        
        #Setting Bounds
        #max K Guess parameters
        # sc = 1e2 #scaling value
        # c = len(self.Stoich.iloc[0,1:]) - len(self.P) - 1 #Number of columns (i.e no. of surface species being investigated)
        
        if self.CovgDep==True:
            initial_vals = np.concatenate((self.k,self.Coeff.flatten(order='F')))
                
        elif self.CovgDep==False:
            initial_vals = self.k
                
        params, params_covariance = optimize.curve_fit(self.covg_func, x_values, y_values
                                                    ,method =method, bounds=(0,inf), maxfev=maxfev, xtol=xtol, ftol=ftol
                                                    ,p0=initial_vals)
        return params, params_covariance
    #------------------------------------------------------------------------------------------------------------------------------ 
    def minimizer_fit_func(self,method,gtol,maxfun,maxiter):
        values = self.normalize()

        x_values = values[:,0]
        y_values = np.reshape(values[:,1:],values[:,1:].size)
        
        #Setting Bounds
        #max K Guess parameters
        sc = 1e3 #scaling value
        c = len(self.Stoich.iloc[0,1:]) - len(self.P) - 1 #Number of columns (i.e no. of surface species being investigated)
             
        if self.CovgDep==True:
            initial_vals = np.concatenate((self.k,self.Coeff.flatten(order='F')))
            mkval = initial_vals*sc #max coeffvals
            n = len(initial_vals)
            lnk = len(self.k)
            bounds = np.empty([c*n,2])
            for i in range(n):
                bounds[i] = (0,mkval[i])  #Rate constants
                bounds[lnk+i] = (-mkval[lnk+i],mkval[lnk+i]) #Rate coefficients
        
        elif self.CovgDep==False:
            initial_vals = self.k
            mkval = initial_vals*sc #max coeffvals
            n = len(initial_vals)
            bounds = np.empty([n,2])
            for i in range(n):
                bounds[i] = (0,mkval[i]) #Rate constants

        result = optimize.minimize(self.cost_func_minimize, initial_vals
                                                    ,method=method, bounds=bounds 
                                                    ,options={'gtol': gtol
                                                              ,'maxfun': maxfun,'disp': False
                                                              ,'maxiter': maxiter})
        
        return result
    #------------------------------------------------------------------------------------------------------------------------------
    def ML_data_gen(self):
        a = len(self.Stoich.iloc[0,1:]) - len(self.P) - 1 #Number of columns (i.e rate coefficients = no. of surface species being investigated)
        b = len(self.k)
        og = self.normalize() #Original input
        
        if self.CovgDep==True:
            rate_cvals = np.concatenate((self.k,self.Coeff.flatten(order='F')))
        elif self.CovgDep==False:
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
            
            if self.CovgDep==False:
                self.MKM.k = Rate_Coeff[i,:]
            elif self.CovgDep==True:
                self.MKM.k = Rate_Coeff[i,:b]
                self.MKM.Coeff = np.reshape(Rate_Coeff[i,b:],(b,a))
                
            sol,solt= self.solve_coverage(t=[0,input_time[-1]],initial_cov=inp_init_covg,Tf_eval=input_time)
            Covg[i,:,:] = sol
            
        return Rate_Coeff,Covg
    #------------------------------------------------------------------------------------------------------------------------------
    def ML_model_predict(self,Covg_fit):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5
                            ,hidden_layer_sizes=(15,)
                            ,random_state=1)
        
        Rate_Coeff,Covg = self.ML_data_gen()
        
        a = np.shape(Covg)[0]
        b = np.shape(Covg)[1]
        c = np.shape(Covg)[2]
        d = np.shape(Rate_Coeff)[0]
        e = np.shape(Rate_Coeff)[1]
        
        print(a)
        print(b)
        print(c)
        print(d)
        print(e)

        le = preprocessing.LabelEncoder()
        le.fit(Rate_Coeff.reshape(-1))
        Rate_Coeff_enc = le.transform(Rate_Coeff.reshape(-1))#.reshape(a,-1)
        l = np.shape(Rate_Coeff_enc)
        print(l)
        clf.fit(Covg.reshape(120,-1),Rate_Coeff_enc)
        
        
        pred = le.transform(Covg_fit.reshape(-1))
        self.fit_params = clf.predict(pred)
        
        return self.fit_params    
    #------------------------------------------------------------------------------------------------------------------------------
    def fitting_rate_param(self,option='cf',plot=False,method_cf='trf',method_min='L-BFGS-B'
                           ,maxfev=1e4,xtol=1e-10,ftol=1e-8,gtol=1e-8,maxfun=1e5,maxiter=5,weight=1e2):

        colmn = len(self.Stoich.iloc[0,1:]) - len(self.P) - 1 #Number of columns (i.e no. of surface species being investigated)
        index = list(string.ascii_lowercase)[:colmn]
        
        og = self.normalize()
            
        if option=='cf':
            blockPrint() #Disable printing
            params, params_covariance = self.curve_fit_func(method=method_cf,maxfev=maxfev,xtol=xtol,ftol=ftol)
    
            x_values = og[:,0] #OG time values
            yfit = self.covg_func(x_values, *params)
            enablePrint() #Re-enable printing
            covg_fit=yfit.reshape(np.shape(og[:,1:]))
            converg = np.sqrt(np.diag(params_covariance))
            
        elif option=='min':
            blockPrint() #Disable printing
            self.min_weight= weight
            result = self.minimizer_fit_func(method=method_min,gtol=gtol,maxfun=maxfun,maxiter=maxiter)
            params = result.x
            
            x_values = og[:,0] #OG time values
            yfit = self.covg_func(x_values, *params)
            enablePrint() #Re-enable printing
            covg_fit=yfit.reshape(np.shape(og[:,1:]))
            
            
            # Finding confidence intervals#--Needs fixing
            # fvec = params
            # jac = np.array([result.jac])
            # print(jac.shape)
            # converg = self.CI95(fvec, jac)
            
        elif option=='ML':
            # blockPrint() #Disable printing
            result=self.ML_model_predict(og[:,1:])
            params=result
            
            x_values = og[:,0] #OG time values
            yfit = self.covg_func(x_values, *params)
            # enablePrint() #Re-enable printing
            covg_fit=yfit.reshape(np.shape(og[:,1:]))
            
        time = og[:,0]
        covg_og = og[:,1:]
        n = len(self.k)
        print('\n \033[1m' + 'Initial guess: \n'+ '\033[0m')
        print('-> Rate Constants:\n',self.k)
        if self.CovgDep==True:
            for i in np.arange(colmn):
                print('-> %s constants:\n'%(str(index[i])),self.Coeff[:,i])
        
        print('\n \033[1m' + 'Final predictions: \n'+ '\033[0m')
        print('-> Rate Constants:\n',params[0:n])
        if self.CovgDep==True:
            for i in np.arange(colmn):
                print('-> %s constants:\n'%(str(index[i])),params[(i+1)*n:(i+2)*n])
        
        # print('\n \033[1m' + 'Confidence Intervals: \n'+ '\033[0m')
        # print('-> Rate Constants:\n',converg[0:n])
        # if self.CovgDep==True:
        #     for i in np.arange(colmn):
        #         print('-> %s constants:\n'%(str(index[i])),converg[(i+1)*n:(i+2)*n])
        
        # Plotting
        if plot==False:    
            return time,covg_og,covg_fit
        elif plot==True:
            self.plotting(time,covg_og,covg_fit,self.label)
            return time,covg_og,covg_fit
        
    #------------------------------------------------------------------------------------------------------------------------------
    #Function responsible for plotting
    #------------------------------------------------------------------------------------------------------------------------------    
    def plotting(self,time,covg_og,covg_fit,label):
        
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
            ax.set_title('Fitting rate parameters')
            
        ax.legend(np.append(lbl_og,lbl_fit),fontsize=10, loc='upper right',facecolor='white', edgecolor ='black', framealpha=1)