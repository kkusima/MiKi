
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

species = open('specnum_output.txt','r').readlines()
TOFofCO2=[]
CO_coverage = []
O_coverage = []
for i in range(2,len(species)):
    ni = float(species[i].split()[-1])
    ni_1 = float(species[i-1].split()[-1])
    ti = float(species[i].split()[2])
    ti_1 = float(species[i-1].split()[2])
    TOFtransient = (ni-ni_1)/(ti-ti_1)
    TOFofCO2.append(TOFtransient)

for i in range(2,len(species)):
    O_i = float(species[i].split()[-6])/(96*96)
    O_coverage.append(O_i)
    
for i in range(2,len(species)):
    CO_i = float(species[i].split()[-5])/(96*96)
    CO_coverage.append(CO_i)
time = []
for i in range(2,len(species)):
    ti = float(species[i].split()[2])
    time.append(ti)
    
plt.rc('font', size=20)
fig,ax1 = plt.subplots(figsize=(20,10))
ax2 = ax1.twinx()
#plt.legend()
plt.xlabel('time')



ax1.plot(time,O_coverage, label = 'O*',color = 'red',linewidth = 1)
ax1.plot(time,CO_coverage, label = 'CO',color = 'green', linewidth = 1)
ax1.legend(loc = 'upper left')
ax1.set_xlabel('time')
ax1.set_ylabel('coverage',labelpad = 20)
ax1.set_ylim(0,2)


ax2.plot(time, TOFofCO2,label = 'TOF of CO2', color = 'blue', linewidth = 4)
ax2.legend(loc = 'upper right')
ax2.set_ylim(0,3000)

plt.ticklabel_format(axis = 'x', style = 'sci',scilimits = (0,0))
ax2.ticklabel_format(axis = 'y', style = 'sci',scilimits = (0,0))
#plt.legend()
ax2.set_ylabel('TOF of CO2', labelpad = 20)
#plt.legend()
plt.savefig('coverage+TOF.png')
