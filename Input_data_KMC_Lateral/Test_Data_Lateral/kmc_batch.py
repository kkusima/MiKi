import os, glob
files = glob.glob('Test_*/*_kmc.csh')
files.sort()

cwd = os.getcwd()

for d in files:
    os.chdir(d.rsplit('/',1)[0]) ## changing directory to where the corresponding directory is
    kmc_file = d.rsplit('/',1)[1]  ## extracting the kmc simulation file name
    os.system('sbatch ' + kmc_file)
    os.chdir(cwd)
