import os, glob
dirs = glob.glob("*/*")
#dirs.sort()

cwd = os.getcwd()

for d in dirs:
    if os.path.isdir(d):
        os.chdir(d)
        os.system('rm *\ 2.*')
        print(d)
        os.chdir(cwd)
