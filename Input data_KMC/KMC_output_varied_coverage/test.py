
# import numpy as np
import glob, os, shutil
# Get the absolute path of the current file
current_file = os.path.abspath(__file__)

# Get the directory path of the current file
current_dir = os.path.dirname(current_file)

#Changing direction to make sure that the output and the os is where this python script is
os.chdir(current_dir)

output_file = 'ML_modelling_output.txt'
ofile = open(output_file,'w')

ofile.write('Testing NEW CHanges')
ofile.write('\n')
import matplotlib.pyplot as plt


# Create a list of data to plot
data = [1, 2, 3, 4, 5]
name = ['one','two','three','four','five']

# Create a loop to create and save each plot
Plot_Folder = 'Plots'
shutil.rmtree(Plot_Folder, ignore_errors=True)
os.mkdir(Plot_Folder)
for i in range(len(data)):
    # Create a figure and plot the data
    fig, ax = plt.subplots()
    ax.plot(data[:i+1])
    
    # Save the plot to a PNG file
    filename = 'plot_{}.png'.format(name[i])
    filepath = os.path.join(Plot_Folder, filename)
    plt.savefig(filepath)
    
    # Close the figure to free up memory
    plt.close(fig)