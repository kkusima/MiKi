import os, glob
dirs = glob.glob('*bar')
dirs.sort()

cwd = os.getcwd()

for d in dirs:
    os.chdir(d)
    os.system('sbatch kmc.csh')
    os.chdir(cwd)
