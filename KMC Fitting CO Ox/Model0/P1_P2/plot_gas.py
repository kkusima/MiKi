
from matplotlib import pyplot
species = open('specnum_output.txt','r').readlines()
TOFofCO2=[]

for i in range(2,len(species)):
    ni = float(species[i].split()[-1])
    ni_1 = float(species[i-1].split()[-1])
    ti = float(species[i].split()[2])
    ti_1 = float(species[i-1].split()[2])
    TOFtransient = (ni-ni_1)/(ti-ti_1)
    TOFofCO2.append(TOFtransient)
    
time = []
for i in range(2,len(species)):
    ti = float(species[i].split()[2])
    time.append(ti)
pyplot.xlabel('time / s')
pyplot.ylabel('TOF of CO2 / (1/s)')    
pyplot.plot(time,TOFofCO2)
pyplot.ylim(0,3000)
pyplot.show()

pyplot.savefig('CO2-TOF.jpg')    
