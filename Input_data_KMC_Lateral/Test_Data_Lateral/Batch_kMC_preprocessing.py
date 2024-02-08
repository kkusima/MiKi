
import os
import glob

files =glob.glob('Test_*')
cwd = os.getcwd()

for i in files:
    os.chdir(i)
    os.getcwd()
    os.system('python KMC_data_preprocessing.py')
    os.chdir(cwd)