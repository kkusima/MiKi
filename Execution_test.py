from main import *
import numpy as np
import matplotlib.pyplot as plt


MKM = MKModel('Atomic.csv','Stoich.csv','Param.csv')

MKM.set_initial_coverages()

MKM.set_rxnconditions(Pr=[1e-8,1e-8,1e-8])

MKM.set_limits_of_integration()

print(MKM.init_cov)

# MKM.solve_coverage(t=[0,6e6])


print(MKM.get_rates())
print('Stoich')
print(MKM.Stoich)
print(len(MKM.Stoich.iloc[0,:]))



print(MKM.set_initial_coverages([0.1,0.2,0]))

print(MKM.Stoich.iloc[0,1])
print('Rates')
R = (MKM.get_ODEs(np.linspace(0,6,num=10),[0,0.1,0.1],coverage=False))
print(R)
print(len(R))
print(MKM.get_rates(cov=[0,0.1,0.1]))
# print(len(R))
print('end')

print(MKM.get_rates())
print(MKM.init_cov)

MKM.set_limits_of_integration()

# sol,solt= MKM.solve_coverage(t=[0,6e6],full_output=True)

# plt.plot(solt,sol[:,0],solt,sol[:,1],solt,sol[:,2],solt,sol[:,3])

# print(np.shape(sol)[0])

print(MKM.get_SS_coverages())

print(MKM.get_SS_rates_reaction())

print(MKM.get_SS_rates_production())


sol,solt = (MKM.dynamic_transient_rates_production(State1=[0.2e-9,2e-6,1e-8],State2=[0.8e-5,2e-6,1e-8],plot=True))
# # 
# plt.plot(solt,sol[:,0],solt,sol[:,1],solt,sol[:,2],solt,sol[:,3],solt,sol[:,4],solt,sol[:,5],solt,sol[:,6])

# s = (5e-2,1e-8,1e-8)
# print(len(s))
print(MKM.Stoich.iloc[:,0])
print(MKM.Atomic.columns.values[1:][0])
print(MKM.label)
# 0.2e-9,2e-6,1e-8
# 0.8e-5,2e-6,1e-8