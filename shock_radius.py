import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#To refresh data:
###On Andes:
#$cd /gpfs/alpine/ast154/scratch/sshanka/carpetx_github/CCSN_12000km_analysis
#$python copy_files.py

###On my laptop
#$cd /home/swapnnil/Desktop/000_carpetX/NERSC_GPU_Hackathon/Balsara_files/test-GRHydroX-GPU-1/CCSN_12000km_analysis
#$rsync -rtv sshanka@andes.olcf.ornl.gov:/gpfs/alpine/ast154/scratch/sshanka/carpetx_github/CCSN_12000km_analysis/ .

hydro_vars = [ "1:iteration", "2:time",	"3:max_shock_radius"]
hydro_vars_x = [ "1:iteration", "2:time",	"3:max_shock_radius_x"]
hydro_vars_y = [ "1:iteration", "2:time",	"3:max_shock_radius_y"]
hydro_vars_z = [ "1:iteration", "2:time",	"3:max_shock_radius_z"]

#------------------------------------------------------------------------------ 
#Determine the latest output number that is already present in out_path (files before that will not be copied)
var_name = "shocktracker-max_shock_radius_x"
os.system("ls scalars/ | grep {} > temp2.txt".format(var_name))
file1 = open('temp2.txt', 'r')
Lines = file1.readlines()
out_list = []
for line in Lines:
    tempvar = int(line.strip().split('-')[3].split('.')[0])
    out_list.append(tempvar)

#os.system("rm temp2.txt")
latest_copied_output = 0
if(len(out_list) > 0):
    latest_copied_output = max(out_list)
#------------------------------------------------------------------------------   
plt.clf()

#TODO: Following parameters control plot characteristics
#------------------------------------------------------------------------------
plot_max_shock_x = True
plot_max_shock_y = True
plot_max_shock_z = True
plot_max_shock = True
display_physical_time = True


#plot_max_shock_x = False
#plot_max_shock_y = False
#plot_max_shock_z = False
plot_max_shock = False

if display_physical_time:
    time_factor = 203.0
    offset = 71928/203.0
    plt.xlim(-0.4, 60)
else:
    time_factor = 1.0 
    offset = 0.0
    plt.xlim(71800, 73500)
#------------------------------------------------------------------------------ 

plt.title("shocktracker::shock_radius")
#plt.ylim(0.0004, 0.0006)

dot_marker_size = 20
ring_marker_size = 30

for output_number in range(0, latest_copied_output+1):
    if plot_max_shock_x:     
        file_name_x = "scalars/shocktracker-max_shock_radius_x-output-{}.tsv".format(str(output_number).zfill(4))
        data_x = pd.read_csv(file_name_x, sep='\t', names=hydro_vars_x, comment="#");
        plt.scatter(data_x["2:time"]/time_factor-offset, 1.477*data_x["3:max_shock_radius_x"], s=dot_marker_size, marker=".", facecolors='r', edgecolors='r')
        
    if plot_max_shock_y:     
        file_name_y = "scalars/shocktracker-max_shock_radius_y-output-{}.tsv".format(str(output_number).zfill(4))
        data_y = pd.read_csv(file_name_y, sep='\t', names=hydro_vars_y, comment="#");
        plt.scatter(data_y["2:time"]/time_factor-offset, 1.477*data_y["3:max_shock_radius_y"], s=ring_marker_size, marker="o", facecolors='none', edgecolors='b')
        
    if plot_max_shock_z:     
        file_name_z = "scalars/shocktracker-max_shock_radius_z-output-{}.tsv".format(str(output_number).zfill(4))
        data_z = pd.read_csv(file_name_z, sep='\t', names=hydro_vars_z, comment="#");
        plt.scatter(data_z["2:time"]/time_factor-offset, 1.477*data_z["3:max_shock_radius_z"], s=dot_marker_size, marker="*", facecolors='green', edgecolors='green')
        
    if plot_max_shock:     
        file_name = "scalars/shocktracker-max_shock_radius-output-{}.tsv".format(str(output_number).zfill(4))
        data = pd.read_csv(file_name, sep='\t', names=hydro_vars, comment="#");
        plt.scatter(data["2:time"]/time_factor-offset, 1.477*data["3:max_shock_radius"], s=ring_marker_size, marker="s", facecolors='none', edgecolors='black')

plt.grid()
plt.xlabel("time [ms]")
plt.ylabel("radius [km]")    
plt.show()



