import MKmodel #import TMKModel #
import numpy as np

# MKmodel.TMKModel(0,0.2,0,Tfinal=3e6,CovgDep=True,P1=1e-8,P2=1e-8,P3=1e-8).plotting().transientcoverages()

# transientratesofprod
# transientcoverages


# a = (MKmodel.TMKModel(0,0.9,0,CovgDep=True).trans_rate_production())
# print(a)#np.shape(a))
# print(np.shape(a))

# trans_coverages
# trans_rate_reaction
# trans_rate_production

#TMKModel.run_inp(x,y,z,Tfinal=None,CovgDep=False,Tfeval=None)

# x = initial CO coverage, y = initial O Coverage, z = initial O2 Coverage
#Tfinal - Final elapsed time of simulation
#CovgDep - Coverage Dependance
#Tfeval - Range of points the ODE will be evaluating at

#------------------------------------------------------------------------------------------------------------------------------  
# a = (MKmodel.SS_PressVar(0,0.2,0,1e-8,1e-8,1e-8,Tfinal=6e6,CovgDep=True).ss_coverage())
# print(a)#np.shape(a))
# print(np.shape(a))

#ss_coverage
#ss_rate_reaction
#ss_rate_production

#------------------------------------------------------------------------------------------------------------------------------  
P1_1 = 0.2e-9
P2_1 = 0.2e-4
P2 = 2e-6
P3 = 2e-6

# a = (MKmodel.Dynamic(0,1,0,P1_1,P2,P3,P2_1,P2,P3,Tfinal=3e6,CovgDep=True).trans_coverages())
# print(a)#np.shape(a))
# print(np.shape(a))


# trans_coverages
# trans_rate_reaction
# trans_rate_production


MKmodel.Dynamic(0,1,0,P1_1,P2,P3,P2_1,P2,P3,Tfinal=3e6,CovgDep=True).plotting().transientratesofprod()

# transientratesofprod
# transientcoverages
