from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
from scipy import optimize


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
        ExpNoCovg = len(self.Stoich.iloc[0,len(self.P)+1:])
        if init==[]: 
            init=np.zeros(ExpNoCovg-1)
            
        if len(init)!=(ExpNoCovg-1):
            raise Exception('Number of coverage entries do not match what is required. %i entries are needed. (Not including number/coverage of empty sites).'%(ExpNoCovg-1))
        else: 
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

        # Plotting
        
        if plot==False:
            return sol,solt
        elif plot==True:
            self.label='coverage'
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
        
        if plot==False:
            return rates_r,covgt
        elif plot==True:
            self.label='rates_p'
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
    
        if plot==False:
            return rates_p,covgt
        elif plot==True:
            self.label='rates_p'
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
    def Dynamic(self,State1=[],State2=[]): #Function used for storing and prompting the input of the two states involved when dynamically switching pressures
        if State1==[]:
            print('\nThe Pressure Input Format:P1,P2,P3,...\n')
            print("Enter the Pressure Conditions of State 1 below:")
            State1_string = input().split(',')
            State1 = [float(x) for x in State1_string]
        if State2==[]:  
            print("Enter the Pressure Conditions of State 2 below:")
            State2_string = input().split(',')
            State2 = [float(x) for x in State2_string]

        self.set_rxnconditions(Pr=State1)
        SS_State1 = self.get_SS_coverages()[:-1]

        return SS_State1,State2
    #------------------------------------------------------------------------------------------------------------------------------
    def dynamic_transient_coverages(self,State1=[],State2=[],plot=False): #Function used for calculating (and plotting) the dynamic transient coverages
        
        SS_State1,State2 = self.Dynamic(State1,State2)
        
        self.set_rxnconditions(Pr=State2)
        sol,solt = self.solve_coverage(initial_cov=SS_State1)
        
        if plot==False:
            return sol,solt
        elif plot==True:
            self.label='coverage'
            self.plotting(sol,solt,self.label)
            return sol,solt
    #------------------------------------------------------------------------------------------------------------------------------
    def dynamic_transient_rates_reaction(self,State1=[],State2=[],plot=False): #Function used for calculating (and plotting) the dynamic transient rates of reaction
        SS_State1,State2 = self.Dynamic(State1,State2)
        
        self.set_rxnconditions(Pr=State2)
        sol,solt = self.solve_rate_reaction(initial_coverage=SS_State1)
        
        if plot==False:
            return sol,solt
        elif plot==True:
            self.label='rates_r'
            self.plotting(sol,solt,self.label)
            return sol,solt
    #------------------------------------------------------------------------------------------------------------------------------
    def dynamic_transient_rates_production(self,State1=[],State2=[],plot=False): #Function used for calculating (and plotting) the dynamic transient rates of production
        SS_State1,State2 = self.Dynamic(State1,State2)
        
        self.set_rxnconditions(Pr=State2)
        sol,solt = self.solve_rate_production(initial_coverage=SS_State1)
        
        if plot==False:
            return sol,solt
        elif plot==True:
            self.label='rates_p'
            self.plotting(sol,solt,self.label)
            return sol,solt
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
            ax.legend(self.Stoich.iloc[:,0])
            ax.set_xlabel('Time, t, [s]')
            ax.set_ylabel(r"Rates of Reaction, $r_i, [TOF]$")
            ax.set_title('Rates of reaction versus Time')
            
        elif label=='coverage':
            ax.legend(self.Atomic.columns.values[1+len(self.P):])
            ax.set_xlabel('Time, t, [s]')
            ax.set_ylabel(r"Coverage, $\theta_i, [ML]$")
            ax.set_title('Coverages versus Time')
            
class MKModel_wCD:#wCD = with Coverage Dependance
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
        ExpNoCovg = len(self.Stoich.iloc[0,len(self.P)+1:])
        if init==[]: 
            init=np.zeros(ExpNoCovg-1)
            
        if len(init)!=(ExpNoCovg-1):
            raise Exception('Number of coverage entries do not match what is required. %i entries are needed. (Not including number/coverage of empty sites).'%(ExpNoCovg-1))
        else: 
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
            K = kref*np.exp(np.sum(np.multiply(Coeff,Theta)))  #/RT lumped into a and b assuming T is constant
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

        # Plotting
        
        if plot==False:
            return sol,solt
        elif plot==True:
            self.label='coverage'
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
        
        if plot==False:
            return rates_r,covgt
        elif plot==True:
            self.label='rates_p'
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
    
        if plot==False:
            return rates_p,covgt
        elif plot==True:
            self.label='rates_p'
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
    def Dynamic(self,State1=[],State2=[]): #Function used for storing and prompting the input of the two states involved when dynamically switching pressures
        if State1==[]:
            print('\nThe Pressure Input Format:P1,P2,P3,...\n')
            print("Enter the Pressure Conditions of State 1 below:")
            State1_string = input().split(',')
            State1 = [float(x) for x in State1_string]
        if State2==[]:  
            print("Enter the Pressure Conditions of State 2 below:")
            State2_string = input().split(',')
            State2 = [float(x) for x in State2_string]

        self.set_rxnconditions(Pr=State1)
        SS_State1 = self.get_SS_coverages()[:-1]

        return SS_State1,State2
    #------------------------------------------------------------------------------------------------------------------------------
    def dynamic_transient_coverages(self,State1=[],State2=[],plot=False): #Function used for calculating (and plotting) the dynamic transient coverages
        
        SS_State1,State2 = self.Dynamic(State1,State2)
        
        self.set_rxnconditions(Pr=State2)
        sol,solt = self.solve_coverage(initial_cov=SS_State1)
        
        if plot==False:
            return sol,solt
        elif plot==True:
            self.label='coverage'
            self.MKM.plotting(sol,solt,self.label)
            return sol,solt
    #------------------------------------------------------------------------------------------------------------------------------
    def dynamic_transient_rates_reaction(self,State1=[],State2=[],plot=False): #Function used for calculating (and plotting) the dynamic transient rates of reaction
        SS_State1,State2 = self.Dynamic(State1,State2)
        
        self.set_rxnconditions(Pr=State2)
        sol,solt = self.solve_rate_reaction(initial_coverage=SS_State1)
        
        if plot==False:
            return sol,solt
        elif plot==True:
            self.label='rates_r'
            self.MKM.plotting(sol,solt,self.label)
            return sol,solt
    #------------------------------------------------------------------------------------------------------------------------------
    def dynamic_transient_rates_production(self,State1=[],State2=[],plot=False): #Function used for calculating (and plotting) the dynamic transient rates of production
        SS_State1,State2 = self.Dynamic(State1,State2)
        
        self.set_rxnconditions(Pr=State2)
        sol,solt = self.solve_rate_production(initial_coverage=SS_State1)
        
        if plot==False:
            return sol,solt
        elif plot==True:
            self.label='rates_p'
            self.MKM.plotting(sol,solt,self.label)
            return sol,solt
#------------------------------------------------------------------------------------------------------------------------------    
# class Fitting:    
    
    
    
    
    
    
    
    