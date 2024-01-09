import glob
import os

## NOTE: Start with 4 , then 3, then 2
num = 4

string = 'Sim_*/*'+str(num)+'.*'
files = glob.glob(string)

# files2 = glob.glob('Sim_*/*2.*')
# files3 = glob.glob('Sim_*/*3.*')
# files4 = glob.glob('Sim_*/*4.*')


for path in files:
    root = path.rsplit(' '+str(num),1)[0] + '*'
    all_list = glob.glob(root)
    for i in all_list:
        if path != i:
            os.system('rm "%s"' % i)

for path in files:
    old = path
    p1 = path.rsplit(' '+str(num),1)[0]
    p2 = path.rsplit(' '+str(num),1)[1]
    os.system('mv "%s" %s%s' % (old,p1,p2))

