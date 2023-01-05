
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from matplotlib.pyplot import figure
#from matplotlib.pyplot import scatter
lattice_length = 96
site_types = 2

    
file1 = open('lattice_output.txt','r')
file2 = open('history_output.txt','r')
file3 = open ('configuration.txt','wb')
sites = lattice_length*lattice_length*site_types
lines1 = file1.readlines()
lines2 = file2.readlines()
num_config = int(lines2[-sites-2].split()[1])
print ('number of configuration is:',num_config)
    #empty plot configuration. X1Y1, X2Y2 are scatters for two sites
X1=[0]
Y1=[0]
for row in range(lattice_length):
    start_x = 0 + row * 1 
    start_y = 0 + row * 1.73
    for column in range(lattice_length):
        x = start_x + column * 2
        y = start_y
        X1.append(x)
        Y1.append(y)
X2=[]
Y2=[]      
for row in range(lattice_length):
    start_x =  1 + row * 1 
    start_y =  1/1.73 + row * 1.73
    for column in range(lattice_length):
        x = start_x + column * 2
        y = start_y
        X2.append(x)
        Y2.append(y)
        
#locating CO and O  
pic_1 = 2
pic_2 = num_config//3
#pic_3 = 2*pic_2
pic_4 = num_config

p_1 = lines2[7+sites+2:7+sites+2+sites]
p_2 = lines2[7+(pic_2-pic_1)*(sites+2):7+(pic_2-pic_1)*(sites+2)+sites]
#p_3 = lines2[7+(pic_3-pic_1)*(sites+2):7+(pic_3-pic_1)*(sites+2)+sites]
p_4 = lines2[-sites-1:-1]

P = [p_1,p_2,p_4]


x=1
m = 0
for p in P:
    m += 1 
    path = os.getcwd()+'/plot' + str(x)
    if not os.path.exists(path):
        os.makedirs(path) 
    index_O = [] #occupied by O*
    index_CO = [] #CO*
    for i in range(len(p)):
        if float(p[i].split()[2])==1:
            index_O.append(i)
    for i in range(len(p)):
        if float(p[i].split()[2])==2:
            index_CO.append(i)

    
    coord_x_fcc_O = []
    coord_y_fcc_O = []
    coord_x_hcp_O = []
    coord_y_hcp_O = []
    coord_x_fcc_CO = []
    coord_y_fcc_CO = []
    coord_x_hcp_CO = []
    coord_y_hcp_CO = []
#O
    for i in index_O:
        a = i//(site_types*lattice_length)
        b = i%(site_types*lattice_length)
        x_start = a*2 
        if b%2 != 0:
            x1 = x_start+(b-1)/2
            y1 = 1.73*((b-1)/2)
            coord_x_fcc_O.append(x1)
            coord_y_fcc_O.append(y1)
        if b%2 == 0:
            if b ==0:
                b = site_types*lattice_length
                x2 = x_start + b/2
                y2 = 1/1.73 + (b/2-1)*1.73
            else:    
                x2 = x_start + b/2
                y2 = 1/1.73 + (b/2-1)*1.73
            coord_x_hcp_O.append(x2)
            coord_y_hcp_O.append(y2)
#CO 
    for i in index_CO:
        a = i//(site_types*lattice_length)
        b = i%(site_types*lattice_length)
        x_start = a*2 
        if b%2 != 0:
            x1 = x_start+(b-1)/2
            y1 = 1.73*((b-1)/2)
            coord_x_fcc_CO.append(x1)
            coord_y_fcc_CO.append(y1)
        if b%2 == 0:
            if b ==0:
                b = site_types*lattice_length
                x2 = x_start + b/2
                y2 = 1/1.73 + (b/2-1)*1.73
            else:    
                x2 = x_start + b/2
                y2 = 1/1.73 + (b/2-1)*1.73
            coord_x_hcp_CO.append(x2)
            coord_y_hcp_CO.append(y2)
    plt.figure(figsize=(40, 20), dpi=40)
    plt.scatter(X1,Y1,c='grey', marker="o",s=100)
    plt.scatter(X2,Y2,c='grey', marker="^",s=100)
    plt.scatter(coord_x_fcc_O,coord_y_fcc_O,c='red', marker="o",s=100)
    plt.scatter(coord_x_hcp_O,coord_y_hcp_O,c='red', marker="^",s=100)
    plt.scatter(coord_x_fcc_CO,coord_y_fcc_CO,c='b', marker="o",s=100)
    plt.scatter(coord_x_hcp_CO,coord_y_hcp_CO,c='b', marker="^",s=100)
    filename = os.path.join(path,'plot' +str (m)+ '.png')
    plt.savefig(filename)  
