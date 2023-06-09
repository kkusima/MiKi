import os, glob
dirs = glob.glob("*/*")
#dirs.sort()

cwd = os.getcwd()

for d in dirs:
    if os.path.isdir(d):
        print(d)
        os.chdir(d)
        os.system('rm *\ 2.*')
        os.chdir(cwd)
