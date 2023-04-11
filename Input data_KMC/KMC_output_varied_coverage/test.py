
# import numpy as np
import glob, os
os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir))#Changes directory back to where this script is

output_file = 'ML_modelling_output.txt'
ofile = open('ML_modelling_output.txt','w')

ofile.write('Testing if it works')