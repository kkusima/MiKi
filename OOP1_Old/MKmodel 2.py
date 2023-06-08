from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import MiKi
from MiKi import Coefficients, ratecoeff
from scipy import optimize

#------------------------------------------------------------------------------------------------------------------------------
def rate_reac(t,u,k,a,b,c,Atomic,Stoich,P1,P2,P3,SS=False):     
    #Function to give the rates of reactions
 
    #---- --- --- --- ---- --- --- ---- ---- ----
    # 1 = CO ; 2 = O ; 3 = O2 ; 4 = *
    #Surface Coverage 
    th1 = u[0] #Theta_CO 
    th2 = u[1] #Theta_O
    th3 = u[2] #Theta_O2
    th4 = u[3] #Theta_*
    THETA = [th1,th2,th3,th4] #Coverages being investigated
    
    Nr = len(Stoich) #Number of rows in your your stoich matrix, i.e (Number of reactions)

    kf = k[0::2] #Pulling out the forward rxn rate constants (::2 means every other value, skip by a step of 2)
    kr = k[1::2] #Pulling out the reverse rxn rate constants 
    
    a_f = a[0::2]
    a_r = a[1::2]
    
    b_f = b[0::2]
    b_r = b[1::2]
    
    c_f = c[0::2]
    c_r = c[1::2]
    
    #Note: # th1 = CO ; th2 = O ; th3 = O2 ; th4 = *
    #for i in range(len(kf)): #Adding Coverage dependence to the rate coefficient
    #    kf[i] = ratecoeff(kf[i],a,b,th1,th2)
    #    kr[i] = ratecoeff(kr[i],a,b,th1,th2)

    r = [None] * Nr  #Empty Vector for holding rate of a specific reaction
    
    #Calculating the rates of reactions:
    for j in np.arange(Nr):   #Looping through the reactions
        matr = [P1,P2,P3]+THETA  #concatenating into the matrix, matr
        fwd = []
        rvs = []
        for i in np.arange(len(Stoich.iloc[0,:])-1):
            if Stoich.iloc[j,i+1]<0: #extracting only forward relevant rate parameters  #forward rxn reactants /encounter probability
                fwd.append(matr[i]**abs(Stoich.iloc[j,i+1]))
                
            if Stoich.iloc[j,i+1]>0: #extracting only reverse relevant rate parameters  #reverse rxn reactants /encounter probability
                rvs.append(matr[i]**abs(Stoich.iloc[j,i+1]))   
                
        r[j] = (ratecoeff(kf[j],a_f[j],b_f[j],c_f[j],th1,th2,th3)*np.prod(fwd)) - (ratecoeff(kr[j],a_r[j],b_r[j],c_r[j],th1,th2,th3)*np.prod(rvs)) #Calculating the rate of reaction
    
    #Steady State implementation:
    msg = 'No error' 
    
    if SS==False:
        return (r,msg)
    
    else:
        r = np.transpose(r)
        lt = len(r[:,0])
        pc = 0.10 #Percent away, to test for steady state
        rend = r[-1,:]
        rend_1 = r[-int(np.round(pc*lt)),:]
        rdiff = np.abs(rend-rend_1)
        if all(x < 1e-8 for x in rdiff)==True:
            return (rend,msg)
        else:
            msg = 'Warning: Difference in a set of last two rates of reactiion terms is NOT less than 1e-8. STEADY STATE MAY NOT HAVE BEEN REACHED. Last terms are returned anyways.'
            #print(msg)
            return (rend,msg)
#------------------------------------------------------------------------------------------------------------------------------
def rate_p(t,u,k,a,b,c,Atomic,Stoich,P1,P2,P3,SS=False):
# #Produces the rates of production for the 4 species

    (r,msg) = rate_reac(t,u,k,a,b,c,Atomic,Stoich,P1,P2,P3,SS)
    
    if msg!='No error':
        print(msg)
        
    D = []      #Empty Vector For holding rate of change of coverage values
    #Differential Equations to calculate the rate of change in coverages
    for i in np.arange(len(Stoich.iloc[:,4:])):
        dsum=0
        for j in np.arange(len(Stoich)):
            
            dsum += Stoich.iloc[j,i+4]*r[j] #Calculating the rate of production of a species i
        
        D.append(dsum)
        
    if SS==False:
        return D  #   CO |  O  |  O2   |  *
    
    else:
        D = np.transpose(D)
        return D[-1,:]
#------------------------------------------------------------------------------------------------------------------------------
def Kinetics(k,a,b,c,Atomic,Stoich,P1,P2,P3,init,Tfinal=None,CovgDep=False,Tfeval=None,SS=False):
    #Getting the time and coverages 

    if Tfinal == None:
        Tfinal = 6e6
    
    Time = np.linspace(0, Tfinal, num=1000)
    
    if Tfeval != None:
        Teval = np.linspace(0, Tfeval, num=1000)
    else:
        Teval = Tfeval
        
    t_span = (Time[0], Time[-1])
    solve = solve_ivp(rate_p,t_span,init, args=(k,a,b,c,Atomic,Stoich,P1,P2,P3),method='BDF', t_eval=Teval, rtol = 1E-6,atol = 1E-8)
    sol = np.transpose(solve.y)
    solt = np.transpose(solve.t)
    
    if SS==False:
        return sol,solt  #   CO |  O  |  O2   |  *
    
    if SS==True:
        lt = len(sol[:,0])
        pc = 0.10 #Percent away, to test for steady state
        solend = sol[-1,:]
        solend_1 = sol[-int(np.round(pc*lt)),:]
        soldiff = np.abs(solend-solend_1)
        if all(x < 1e-2 for x in soldiff)==True:
            return solend,solt[-1]
        else:
            print('Warning: Difference in a set of last two coverage terms is NOT less than 1e-2. STEADY STATE MAY NOT HAVE BEEN REACHED. Last terms are returned anyways.')
            return solend,solt[-1]
#------------------------------------------------------------------------------------------------------------------------------
def massbalance(Atomic,Stoich): #Function to check if mass is balanced
    at_mat = Atomic.iloc[0:,1:]           #The atomic matrix
    err = 0                               #For counting error
    for i in np.arange(len(Stoich)):    
        st_mat = Stoich.iloc[i,1:]        #The stoichiometric matrix
        res = np.dot(at_mat,st_mat)       #Performing the matrix product for every reaction i
        if any(a != 0 for a in res):      #Verifies that the matrix product returns 0s (i.e mass is balanced)
            text = "Mass is not conserved in reaction %i. \n ... Check and correct the Atomic and/or Stoichiometric Matrices"%(i+1)
            #print(text)
            err +=1
            raise Exception(text,'\n')
        elif (i == len(Stoich)-1 and err==0):
            text = "Mass is conserved."
            #print(text)
            return print('') #print(text,'\n')
#------------------------------------------------------------------------------------------------------------------------------     
def coverage_check(TCO_in,TO_in,TO2_in,TE_in) :
       if ((TCO_in < 0 or TO_in < 0 or TO2_in < 0 or TE_in < 0  or TCO_in > 1 or TO_in > 1 or TO2_in > 1 or TE_in > 1)) :
           raise Exception('Error: The initial coverages entered are not valid. \n Please double check the initial coverages entered and make the necessary corrections')
#------------------------------------------------------------------------------------------------------------------------------
class TMKModel: #Transient Microkinetic Model (constant pressure as inputted in Param)
    def __init__(self,x,y,z,Tfinal=None,CovgDep=False,Tfeval=None,P1=None,P2=None,P3=None):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        self.x = x
        self.y = y
        self.z = z
        self.Tfinal = Tfinal
        self.CovgDep = CovgDep
        self.Tfeval = Tfeval   
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3         
        
        #Checking coverage
        self.TCO_in = self.x
        self.TO_in = self.y
        self.TO2_in = self.z
        self.TE_in = 1 - self.TCO_in-self.TO_in-self.TO2_in
        
        coverage_check(self.TCO_in,self.TO_in,self.TO2_in,self.TE_in)
        
        #Checking Mass Balance
        massbalance(MiKi.Atomic, MiKi.Stoich)
        
        #Relevant Coefficients:
        self.Atomic = MiKi.Atomic
        self.Stoich = MiKi.Stoich
        self.Param = MiKi.Param 
        
        self.k = Coefficients(self.Param).kextract()
        
        
        if self.CovgDep==False:
            self.a = 0*np.ones(len(self.k))
            self.b = 0*np.ones(len(self.k))
            self.c = 0*np.ones(len(self.k))
            
        else:
            self.a = Coefficients(self.Param).aextract()
            self.b = Coefficients(self.Param).bextract()
            self.c = Coefficients(self.Param).cextract() 
            
        Press = Coefficients(self.Param).Pextract() #Inputted pressure
        
        if self.P1 == None:
            self.P1 = Press[0] #P_CO
        if self.P2 == None:
            self.P2 = Press[1] #P_O2
        if self.P3 == None:
            self.P3 = Press[2] #P_CO2    
    #------------------------------------------------------------------------------------------------------------------------------         
    def plotting(self): ## Allowing the outer class to access inner plot class
        return self.plots(self)
    #------------------------------------------------------------------------------------------------------------------------------    
    def trans_coverages(self):
    #Function that gives transient coverage given the class inputs(attributes)    
    #Function for calculating and plotting the transient coverage for an initially empty surface coverage

        #     #init      CO | O  | O2   | *
        self.init = [self.TCO_in,self.TO_in,self.TO2_in,self.TE_in]  #initial values
            
            
        solve = Kinetics(self.k,self.a,self.b,self.c,self.Atomic,self.Stoich,self.P1,self.P2,self.P3,self.init,self.Tfinal,self.CovgDep,self.Tfeval)
        
        solv = solve[0]
        time_sol = solve[1]

        C_CO = solv[:,0]
        C_O = solv[:,1]
        C_O2 = solv[:,2]
        C_E = solv[:,3]
        
       #Transient Coverage with intial surface coverage

        return [time_sol,C_CO,C_O,C_O2,C_E]
    #------------------------------------------------------------------------------------------------------------------------------
    def trans_rate_reaction(self):
        #Function that gives transient rates of reaction given the class inputs(attributes)
        solve = self.trans_coverages()
        #Note
        #t = solve[0]
        #u = solve[1:]   
 
        rate_reaction = rate_reac(solve[0],solve[1:],self.k,self.a,self.b,self.c,self.Atomic,self.Stoich,self.P1,self.P2,self.P3)
       
        return np.append([solve[0]], rate_reaction, axis=0) #Adding time entries to the rate_reaction vector
    #------------------------------------------------------------------------------------------------------------------------------
    def trans_rate_production(self):
    #Function that gives transient rates of production given the class inputs(attributes)
    #Calculating the rates of production of surface species:
        solve = self.trans_coverages()

        #Note
        #t = solve[0]
        #u = solve[1:]  
        
        rate_prod = rate_p(solve[0],solve[1:],self.k,self.a,self.b,self.c,self.Atomic,self.Stoich,self.P1,self.P2,self.P3) #transient rates of production of surface species
        
        #solving for transient rates of reaction:
        rate_r = self.trans_rate_reaction()
        
        rate_p_CO = -1* rate_r[1]
        rate_p_O2 = -1* rate_r[2]
        rate_p_C02 = rate_r[4]
                    
        rates_vect = [solve[0],rate_prod[0],rate_prod[1],rate_prod[2],rate_prod[3],rate_p_CO,rate_p_O2,rate_p_C02]
        
        return rates_vect
    #------------------------------------------------------------------------------------------------------------------------------
    class plots():
        ## instantiating for the 'Inner' class #connecting up and down i.e outer with inner
        ##self.outer will carry instances from the outer class
        def __init__(self, outer):
            self.outer=outer
        #------------------------------------------------------------------------------------------------------------------------------    
        def transientcoverages(self):
            cov = self.outer.trans_coverages()
            time_sol = cov[0]
            C_CO = cov[1]
            C_O = cov[2]
            C_O2 = cov[3]
            C_E = cov[4]
            
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(time_sol, C_CO,'r-', label=r'$CO^*$')        
            ax.plot(time_sol, C_O,'g-', label=r'$O^*$') 
            ax.plot(time_sol, C_O2, 'b-', label=r'$O^*_2$') 
            ax.plot(time_sol, C_E, 'k-', label='*') 
            
            textstr = '\n'.join((
            r'Initial $\theta_i:$',
            r'$\theta_{{CO}}(0)=%.2f$' % (self.outer.TCO_in),
            r'$\theta_{{O}}(0)=%.2f$' % (self.outer.TO_in),
            r'$\theta_{{O_2}}(0)=%.2f$' % (self.outer.TO2_in),
            r'${{\theta^*}}(0)=%.2f$' % (self.outer.TE_in)))
            props = dict(boxstyle='round', facecolor='wheat', alpha=1)
            ax.text(0.62,0.96, textstr, transform=ax.transAxes, fontsize=9,linespacing=0.8,
        verticalalignment='top', bbox=props)
            
            ax.set_xlabel('Time, t, [s]')
            ax.set_ylabel(r"Coverage, $\theta_i, [ML]$")
            ax.legend(fontsize=10, loc='upper right',facecolor='white', edgecolor ='black', framealpha=1)#.set_draggable(state=True,update=('loc'))
            ax.set_title('Coverages versus Time')
        #------------------------------------------------------------------------------------------------------------------------------    
        def transientratesofprod(self):
            rate = self.outer.trans_rate_production()

            time_sol = rate[0]
            R_CO = rate[1]
            R_O = rate[2]
            R_O2 = rate[3]
            R_E = rate[4]
            R_g_CO = rate[5]
            R_g_O2 = rate[6]
            R_g_CO2 = rate[7]
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(time_sol, R_CO,'r-', label=r'$CO^*$')        
            ax.plot(time_sol, R_O,'g-', label=r'$O^*$') 
            ax.plot(time_sol, R_O2, 'b-', label=r'$O^*_2$') 
            ax.plot(time_sol, R_E, 'k-', label='*') 
            ax.plot(time_sol, R_g_CO, 'y-', label='CO') 
            ax.plot(time_sol, R_g_O2, 'm-', label=r'$O_2$') 
            ax.plot(time_sol, R_g_CO2, 'c-', label=r'$CO_2$') 
            
            textstr = '\n'.join((
            r'Initial $\theta_i:$',
            r'$\theta_{{CO}}(0)=%.2f$' % (self.outer.TCO_in),
            r'$\theta_{{O}}(0)=%.2f$' % (self.outer.TO_in),
            r'$\theta_{{O_2}}(0)=%.2f$' % (self.outer.TO2_in),
            r'${{\theta^*}}(0)=%.2f$' % (self.outer.TE_in)))
            props = dict(boxstyle='round', facecolor='wheat', alpha=1)
            ax.text(0.62,0.96, textstr, transform=ax.transAxes, fontsize=9,linespacing=0.8,
        verticalalignment='top', bbox=props)
            
            ax.set_xlabel('Time, t, [s]')
            ax.set_ylabel(r"Rates of Production, $R_i, [TOF]$")
            ax.legend(fontsize=10, loc='upper right',facecolor='white', edgecolor ='black', framealpha=1)#.set_draggable(state=True,update=('loc'))
            ax.set_title('Rate versus Time')
        #------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------           
class SS_PressVar:
    def __init__(self,x,y,z,P1,P2,P3,Tfinal=None,CovgDep=False,Tfeval=None,SS=True):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        self.x = x
        self.y = y
        self.z = z
        self.Tfinal = Tfinal
        self.CovgDep = CovgDep
        self.Tfeval = Tfeval   
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3         
        
        #Steady State Declaration
        self.SS = SS
        
        #Checking coverage
        self.TCO_in = self.x
        self.TO_in = self.y
        self.TO2_in = self.z
        self.TE_in = 1 - self.TCO_in-self.TO_in-self.TO2_in
        
        coverage_check(self.TCO_in,self.TO_in,self.TO2_in,self.TE_in)
        
        #Checking Mass Balance
        massbalance(MiKi.Atomic, MiKi.Stoich)
        
        #Relevant Coefficients:
        self.Atomic = MiKi.Atomic
        self.Stoich = MiKi.Stoich
        self.Param = MiKi.Param 
        
        self.k = Coefficients(self.Param).kextract()
        
        
        if self.CovgDep==False:
            self.a = 0*np.ones(len(self.k))
            self.b = 0*np.ones(len(self.k))
            self.c = 0*np.ones(len(self.k))
            
        else:
            self.a = Coefficients(self.Param).aextract()
            self.b = Coefficients(self.Param).bextract()
            self.c = Coefficients(self.Param).cextract() 
            
        Press = Coefficients(self.Param).Pextract() #Inputted pressure
        
        if self.P1 == None:
            self.P1 = Press[0] #P_CO
        if self.P2 == None:
            self.P2 = Press[1] #P_O2
        if self.P3 == None:
            self.P3 = Press[2] #P_CO2
    #------------------------------------------------------------------------------------------------------------------------------    
    def ss_coverage(self):
    #Function that gives steady state coverage given the class inputs(attributes)    
    #Function for calculating and plotting the transient coverage for an initially empty surface coverage
        #     #init      CO | O  | O2   | *
        self.init = [self.TCO_in,self.TO_in,self.TO2_in,self.TE_in]  #initial values
        solve = Kinetics(self.k,self.a,self.b,self.c,self.Atomic,self.Stoich,self.P1,self.P2,self.P3,self.init,self.Tfinal,self.CovgDep,self.Tfeval,self.SS)
        return solve[0] #returns coverages only   
    
    def ss_rate_reaction(self):
        solve = TMKModel.trans_coverages(self)
        #Note
        #t = solve[0]
        #u = solve[1:] 
        rate_reaction = rate_reac(solve[0],solve[1:],self.k,self.a,self.b,self.c,self.Atomic,self.Stoich,self.P1,self.P2,self.P3,self.SS)
        return rate_reaction
    
    def ss_rate_production(self):
        solve = TMKModel.trans_coverages(self)
        #Note
        #t = solve[0]
        #u = solve[1:]  
        rate_prod = rate_p(solve[0],solve[1:],self.k,self.a,self.b,self.c,self.Atomic,self.Stoich,self.P1,self.P2,self.P3,self.SS) 
        
        #solving for steady state rates of reaction:
        rate_r = self.ss_rate_reaction()
        
        rate_p_CO = -1* rate_r[0]
        rate_p_O2 = -1* rate_r[1]
        rate_p_C02 = rate_r[3]
                    
        rates_vect = [rate_prod[0],rate_prod[1],rate_prod[2],rate_prod[3],rate_p_CO,rate_p_O2,rate_p_C02]
        
        return rates_vect
#------------------------------------------------------------------------------------------------------------------------------        
class Dynamic:        
    def __init__(self,x,y,z,P1_1,P1_2,P1_3,P2_1,P2_2,P2_3,Tfinal=None,CovgDep=False,Tfeval=None):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        self.x = x
        self.y = y
        self.z = z
        self.P1_1 = P1_1
        self.P1_2 = P1_2
        self.P1_3 = P1_3
        self.P2_1 = P2_1
        self.P2_2 = P2_2
        self.P2_3 = P2_3
        self.Tfinal = Tfinal
        self.CovgDep = CovgDep
        self.Tfeval = Tfeval       
        
        #Checking coverage
        self.TCO_in = self.x
        self.TO_in = self.y
        self.TO2_in = self.z
        self.TE_in = 1 - self.TCO_in-self.TO_in-self.TO2_in
        
        coverage_check(self.TCO_in,self.TO_in,self.TO2_in,self.TE_in)
        
        #Checking Mass Balance
        massbalance(MiKi.Atomic, MiKi.Stoich)
        
        #Relevant Coefficients:
        self.Atomic = MiKi.Atomic
        self.Stoich = MiKi.Stoich
        self.Param = MiKi.Param 
        
        self.k = Coefficients(self.Param).kextract()
        
        if self.CovgDep==False:
            self.a = 0*np.ones(len(self.k))
            self.b = 0*np.ones(len(self.k))
            self.c = 0*np.ones(len(self.k))
            
        else:
            self.a = Coefficients(self.Param).aextract()
            self.b = Coefficients(self.Param).bextract()
            self.c = Coefficients(self.Param).cextract()      
    #------------------------------------------------------------------------------------------------------------------------------         
    def plotting(self): ## Allowing the outer class to access inner plot class
        return self.plots(self)
    #------------------------------------------------------------------------------------------------------------------------------    
    def trans_coverages(self):
        #     #init      CO | O  | O2   | *
        self.init = [self.TCO_in,self.TO_in,self.TO2_in,self.TE_in]  #initial values
        #SS Coverages for State 1    
        S1_solv = SS_PressVar(self.TCO_in,self.TO_in,self.TO2_in,self.P1_1,self.P1_2,self.P1_3,self.Tfinal,self.CovgDep,self.Tfeval,True).ss_coverage() 
        
        #Trans Coverages for State 2
        S2_solve = TMKModel(S1_solv[0],S1_solv[1],S1_solv[2],self.Tfinal,self.CovgDep,self.Tfeval,self.P2_1,self.P2_2,self.P2_3).trans_coverages() #SS Coverages for State 2
        return S2_solve
    #------------------------------------------------------------------------------------------------------------------------------
    def trans_rate_reaction(self):
        #Function that gives transient rates of reaction given the class inputs(attributes)
        solve = self.trans_coverages()
        #Note
        #t = solve[0]
        #u = solve[1:]   
 
        rate_reaction = rate_reac(solve[0],solve[1:],self.k,self.a,self.b,self.c,self.Atomic,self.Stoich,self.P2_1,self.P2_2,self.P2_3)
       
        return np.append([solve[0]], rate_reaction, axis=0) #Adding time entries to the rate_reaction vector
    #------------------------------------------------------------------------------------------------------------------------------
    def trans_rate_production(self):
    #Function that gives transient rates of production given the class inputs(attributes)
    #Calculating the rates of production of surface species:
        solve = self.trans_coverages()
        
        #Note
        #t = solve[0]
        #u = solve[1:]  

        rate_prod = rate_p(solve[0],solve[1:],self.k,self.a,self.b,self.c,self.Atomic,self.Stoich,self.P2_1,self.P2_2,self.P2_3) #transient rates of production of surface species
        
        #solving for transient rates of reaction:
        rate_r = self.trans_rate_reaction()
        
        rate_p_CO = -1* rate_r[1]
        rate_p_O2 = -1* rate_r[2]
        rate_p_C02 = rate_r[4]
                    
        rates_vect = [solve[0],rate_prod[0],rate_prod[1],rate_prod[2],rate_prod[3],rate_p_CO,rate_p_O2,rate_p_C02]
        
        return rates_vect
    #------------------------------------------------------------------------------------------------------------------------------    
    class plots():
        ## instantiating for the 'Inner' class #connecting up and down i.e outer with inner
        ##self.outer will carry instances from the outer class
        def __init__(self, outer):
            self.outer=outer
        #------------------------------------------------------------------------------------------------------------------------------    
        def transientcoverages(self):
           cov = self.outer.trans_coverages()
           time_sol = cov[0]
           C_CO = cov[1]
           C_O = cov[2]
           C_O2 = cov[3]
           C_E = cov[4]
           
           fig = plt.figure()
           ax = fig.add_subplot(111)
           ax.plot(time_sol, C_CO,'r-', label=r'$CO^*$')        
           ax.plot(time_sol, C_O,'g-', label=r'$O^*$') 
           ax.plot(time_sol, C_O2, 'b-', label=r'$O^*_2$') 
           ax.plot(time_sol, C_E, 'k-', label='*') 
           
           textstr = '\n'.join((
           r'State 1:',
           r'$P_{{CO}}={:.1e}$'.format(self.outer.P1_1),
           r'$P_{{O}}={:.1e}$'.format(self.outer.P1_2),
           r'$P_{{O_2}}={:.1e}$'.format(self.outer.P1_3),
           '\n State 2:',
           r'$P_{{CO}}={:.1e}$'.format(self.outer.P2_1),
           r'$P_{{O}}={:.1e}$'.format(self.outer.P2_2),
           r'$P_{{O_2}}={:.1e}$'.format(self.outer.P2_3)))
           props = dict(boxstyle='round', facecolor='wheat', alpha=1)
           ax.text(0.58,0.985, textstr, transform=ax.transAxes, fontsize=9,linespacing=0.8,
       verticalalignment='top', bbox=props)
           
           ax.set_xlabel('Time, t, [s]')
           ax.set_ylabel(r"Coverage, $\theta_i, [ML]$")
           ax.legend(fontsize=10, loc='upper right',facecolor='white', edgecolor ='black', framealpha=1)#.set_draggable(state=True,update=('loc'))
           ax.set_title('Coverages versus Time')
       #------------------------------------------------------------------------------------------------------------------------------    
        def transientratesofprod(self):
            rate = self.outer.trans_rate_production()
            
            time_sol = rate[0]
            R_CO = rate[1]
            R_O = rate[2]
            R_O2 = rate[3]
            R_E = rate[4]
            R_g_CO = rate[5]
            R_g_O2 = rate[6]
            R_g_CO2 = rate[7]
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(time_sol, R_CO,'r-', label=r'$CO^*$')        
            ax.plot(time_sol, R_O,'g-', label=r'$O^*$') 
            ax.plot(time_sol, R_O2, 'b-', label=r'$O^*_2$') 
            ax.plot(time_sol, R_E, 'k-', label='*') 
            ax.plot(time_sol, R_g_CO, 'y-', label='CO') 
            ax.plot(time_sol, R_g_O2, 'm-', label=r'$O_2$') 
            ax.plot(time_sol, R_g_CO2, 'c-', label=r'$CO_2$') 
            
            textstr = '\n'.join((
             r'State 1:',
             r'$P_{{CO}}={:.1e}$'.format(self.outer.P1_1),
             r'$P_{{O}}={:.1e}$'.format(self.outer.P1_2),
             r'$P_{{O_2}}={:.1e}$'.format(self.outer.P1_3),
             '\n State 2:',
             r'$P_{{CO}}={:.1e}$'.format(self.outer.P2_1),
             r'$P_{{O}}={:.1e}$'.format(self.outer.P2_2),
             r'$P_{{O_2}}={:.1e}$'.format(self.outer.P2_3)))
            props = dict(boxstyle='round', facecolor='wheat', alpha=1)
            ax.text(0.58,0.985, textstr, transform=ax.transAxes, fontsize=9,linespacing=0.8,
            verticalalignment='top', bbox=props)
            
            ax.set_xlabel('Time, t, [s]')
            ax.set_ylabel(r"Rates of Production, $R_i, [TOF]$")
            ax.legend(fontsize=10, loc='upper right',facecolor='white', edgecolor ='black', framealpha=1)#.set_draggable(state=True,update=('loc'))
            ax.set_title('Rate versus Time') 
        #------------------------------------------------------------------------------------------------------------------------------    