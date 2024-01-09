import os, glob
dirs = glob.glob('Sim_*')
dirs.sort()

cwd = os.getcwd()

for d in dirs:
    os.chdir(d)
    
    os.system('sbatch kmc.csh')
    os.chdir(cwd)
