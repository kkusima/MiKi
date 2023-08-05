#must specify column number and number of instances to average over

import os, glob
dirs = glob.glob('*bar')
dirs.sort()

cwd = os.getcwd()
print(cwd)

import numpy as np
import pandas as pd

def check_SS(trans_vec,tol=0.1,feature='coverage'): #Function for checking if steady state has been reached

        #trans_vector=transient vector #tol=tolerance value i.e what percent distance is between end and end_prev (0.1 means that end_prev is a value 10% away from end)

        length = np.shape(trans_vec)[0]

        end = trans_vec[-1]
        
        n_ss = int(np.round(length*tol))

        end_prev = trans_vec[-(n_ss)]

        steady_diff = np.abs(end-end_prev)
        
        

        

        msg='Steady State Reached'

        if feature=='coverage':

            if steady_diff < 40:

                return (end,msg,n_ss)

            else:

                msg = 'Warning: STEADY STATE MAY NOT HAVE BEEN REACHED. Difference in a set of last two coverage terms is NOT less than 1e-2.Last terms are returned anyways.'

                return (end,msg,n_ss)

#         elif feature=='rates_reaction':

#             if all(x < 1e-7 for x in steady_diff):

#                 return (end,msg)

#             else:

#                 msg = 'Warning: STEADY STATE MAY NOT HAVE BEEN REACHED. Difference in a set of last two rates of reaction terms is NOT less than 1e-7. Last terms are returned anyways.'

#                 return (end,msg)

#         elif feature=='rates_production':

#             if all(x < 1e-7 for x in steady_diff):

#                 return (end,msg)

#             else:

#                 msg = 'Warning: STEADY STATE MAY NOT HAVE BEEN REACHED. Difference in a set of last two rates of production terms is NOT less than 1e-7. Last terms are returned anyways.'

#                 return (end,msg)

for i in dirs:
    os.chdir(i)
    file=open('specnum_output.txt','r').readlines() #Reading in the relevant file
    b=[]
    for j in np.arange(len(file)): 
        b.append(file[j].split())                   #Dividing the rows into columns
    o = pd.DataFrame(data=b)                        #Final output
    vec = o.loc[1:,6].astype(int).to_list()
    (end,msg,n_ss) = check_SS(vec)                            #specify the number of datapoints here, column 6 is CO* and column 5 is O*
    t_avg = sum(list(vec[-n_ss:]))/n_ss
    os.chdir(cwd)
    with open('t_avg.txt','a') as f:
        print(i[:-3],t_avg,file=f)
        print(i,msg)
        