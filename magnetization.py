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


hydro_vars = ["1:iteration", "2:time", "3:grhydrox::magnetization.min", "4:grhydrox::magnetization.max", "5:grhydrox::magnetization.sum", "6:grhydrox::magnetization.avg", "7:grhydrox::magnetization.stddev", "8:grhydrox::magnetization.volume", "9:grhydrox::magnetization.L1norm", "10:grhydrox::magnetization.L2norm", "11:grhydrox::magnetization.maxabs", "12:grhydrox::magnetization.minloc[0]", "13:grhydrox::magnetization.minloc[1]", "14:grhydrox::magnetization.minloc[2]", "15:grhydrox::magnetization.maxloc[0]", "16:grhydrox::magnetization.maxloc[1]", "17:grhydrox::magnetization.maxloc[2]"]

#------------------------------------------------------------------------------ 
#Determine the latest output number that is already present in out_path (files before that will not be copied)
var_name = "grhydrox-magnetization"
os.system("ls norms/ | grep {} > temp2.txt".format(var_name))
file1 = open('temp2.txt', 'r')
Lines = file1.readlines()
out_list = []
for line in Lines:
    tempvar = int(line.strip().split('-')[3].split('.')[0])
    out_list.append(tempvar)

latest_copied_output = 0
if(len(out_list) > 0):
    latest_copied_output = max(out_list)
#------------------------------------------------------------------------------   
plt.clf()

#TODO: Following parameters control plot characteristics
#------------------------------------------------------------------------------
#Do we want to see separation of different checkpoints?
separate_chkpts = False #True
display_physical_time = True

if display_physical_time:
    time_factor = 203.0
    offset = 71928/203.0
    #plt.xlim(-0.4, 150)
    plt.xlim(130, 160)
else:
    time_factor = 1.0 
    offset = 0.0
    plt.xlim(71800, 73500)
#------------------------------------------------------------------------------ 

plt.title("{}".format(var_name))
plt.ylim(0.0, 30)


ring_marker_size = 40
dot_marker_size = 20

plt.axhline(y = 50, color = 'black', linestyle = '--')
plt.axhline(y = 25, color = 'black', linestyle = '--')
plt.axhline(y = 10, color = 'black', linestyle = '--')

style = False #Always start with false so that dot is plotted first, especially when separate_chkpts=False
for output_number in range(59, latest_copied_output+1):     
    file_name = "norms/{}-output-{}.tsv".format(var_name, str(output_number).zfill(4))
    data = pd.read_csv(file_name, sep='\t', names=hydro_vars, comment="#");
    if style:  
        plt.scatter(data["2:time"]/time_factor-offset, data["4:grhydrox::magnetization.max"], s=ring_marker_size, marker="o", facecolors='none', edgecolors='b')
    else:
        #plt.scatter(data["2:time"]/time_factor-offset, data["4:grhydrox::magnetization.max"], s=dot_marker_size, marker=".", facecolors='r', edgecolors='r')
        plt.plot(data["2:time"]/time_factor-offset, data["4:grhydrox::magnetization.max"], marker=".", color='r')
        #plt.semilogy(data["2:time"]/time_factor-offset, data["4:grhydrox::magnetization.max"], color='r')
    #Only toggle styles if we want to separate checkpoints
    if separate_chkpts:
        style = not style


plt.show()



