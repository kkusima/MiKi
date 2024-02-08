import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

Test_Name = os.getcwd().rsplit('/',1)[1]

#%matplotlib notebook
file=open('specnum_output.txt','r').readlines() #Reading in the relevant file
b=[]
for i in np.arange(len(file)): 
    b.append(file[i].split())                   #Dividing the rows into columns
o = pd.DataFrame(data=b)                        #Final output
print(o.head(4))
print(len(o))


#Extracting Number of Sites from the general_output file:
inp=open('general_output.txt','r').readlines()
for i in np.arange(len(inp)): 
    if 'Total number of lattice sites:' in inp[i]:
        val = i  #Line in text file where sentence is present

sites = int(inp[val][34:])
print('\n Number of sites:', sites)


#### Finding number of surface species

headings = (o.iloc[0,:])
n_ss = sum('*' in s for s in headings) #Number of surface species
print('Number of surface species:',n_ss)

#### Finding number of gas species
n_gs = len(headings)-5-n_ss
print('Number of gas species:',n_gs)

#### Adding column to calculate number of empty sites
n_c=(len(o.iloc[0,:])) #number of current columns
o[n_c]=" "           #Creating new empty column 
o.iloc[0,n_c]="*"    #Labelling the new empty column 

print(o.head(4))
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
    
print(o.head(4))

### Surface Species (COVERAGES): 

Sspecies = []
for i in range(n_ss):
    Sspecies.append(5+i) 
Sspecies.append(len(o.iloc[1,:])-1)#Including empty sites
    
print('Column Location of Surface Species',Sspecies) #in this eg, # 5 = O* ; 6 = CO* ; 7 = O2* ; 8 = O2 ; 9 = CO ; 10 = CO2 ; 11 = * {As seen from dataset}

#Calculating itme:
Gtime = o[2][1:].astype(float) 
#Calculating coverages:
Scoverages = np.empty([len(o.iloc[:,1])-1,len(Sspecies)])
for i in range(len(Scoverages[1,:])):
    Scoverages[:,i] = o[Sspecies[i]][1:].astype(float)/sites

#Plotting of effects of Time in seconds -> o[2]:
plt.figure()
for i in range(len(Sspecies)):
    #Plotting Time = x ; Coverage of species i = y
    plt.plot(Gtime,Scoverages[:,i],label=o.iloc[0,Sspecies[i]]) 

plt.legend(fontsize=15, loc='best')
plt.xlabel((r'Time (s)'),size = '15.0')
plt.ylabel('Coverage (ML)',size = '15.0')
plt.title('kMC_Coverage_'+Test_Name)
plt.savefig('kMC_Coverages_'+Test_Name+'.png')
#plt.show()


### GAS SPECIES (No. of Molecules):

Gspecies = []
for i in range(n_gs):
    Gspecies.append(5+n_ss+i) 
    
print('Column Location of Gas Species',Gspecies) #in this eg, # 5 = O* ; 6 = CO* ; 7 = O2* ; 8 = O2 ; 9 = CO ; 10 = CO2 ; 11 = * {As seen from dataset}

#Calculating itme:
Gtime = o[2][1:].astype(float) 
#Extracting the number of gas species molecules:
Gnmol = np.empty([len(o.iloc[:,1])-1,len(Gspecies)])
for i in range(len(Gnmol[1,:])):
    Gnmol[:,i] = o[Gspecies[i]][1:].astype(float) #NEGATIVE INDICATES REACTANT SPECIES (Helpful for calculating rate of consumption)

#Plotting of effects of Time in seconds -> o[2]:
plt.figure()
for i in range(len(Gspecies)):
    #Plotting Time = x ; No of molecules of gas species i = y
    plt.plot(Gtime,Gnmol[:,i],label=o.iloc[0,Gspecies[i]]) 

plt.legend(fontsize=15, loc='best')
plt.xlabel((r'Time (s)'),size = '15.0')
plt.ylabel('Number of Gas species molcs.',size = '15.0')
#plt.show()

### GAS SPECIES ([TOF](https://zacros.org/files/kmc_workshop/Zacros_Tutorial_01.pdf)):

### Calculating the instantaneous rates of profuction (i.e grad/sites)
TOF_GS = np.empty([len(o.iloc[:,1])-1,len(Gspecies)]) #initializing an array of instantaneous TOFs for gaseous species
# grads = np.empty([len(o.iloc[:,1])-1,1])
for i in np.arange(len(Gspecies)):
    grads = np.gradient(Gnmol[:,i],Gtime,edge_order=2)
    TOF_GS[:,i] = grads/sites


#Creating output datframe
Kinetic_Info = pd.DataFrame(np.array(Gtime), columns=['Time'])
#Appending surface coverages
for i in np.arange(np.shape(Scoverages)[1]):
    Kinetic_Info[o.iloc[0,Sspecies[i]]] = pd.Series(Scoverages[:,i])
#Appending gas species Turn over frequencies
for i in np.arange(len(Gspecies)):
    Kinetic_Info['R_'+o.iloc[0,Gspecies[i]]] = pd.Series(TOF_GS[:,i])    

Kinetic_Info    



File_Name = 'KMC_NonDynamic_Data_iCovg_iRates_'+Test_Name+'.csv'

Kinetic_Info.to_csv(File_Name,index=False)

plt.figure()
plt.plot(Kinetic_Info.values[:,0], Kinetic_Info.values[:,5],'r*', label='CO_kMC')        
plt.plot(Kinetic_Info.values[:,0], Kinetic_Info.values[:,6],'g*', label='O2_kMC') 
plt.plot(Kinetic_Info.values[:,0], Kinetic_Info.values[:,7], 'b*', label='CO2_kMC') 
plt.xlabel('Time, s')
plt.ylabel("TOF")
plt.title('TOF_kMC'+Test_Name)
plt.legend(fontsize=5, loc='best')
plt.savefig('kMC_TOF_'+Test_Name+'.png')
#plt.show()