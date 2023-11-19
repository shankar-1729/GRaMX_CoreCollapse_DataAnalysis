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


hydro_vars = ["1:iteration", "2:time", "3:admbase::alp.min", "4:admbase::alp.max", "5:grhydrox::plasma_beta.sum", "6:grhydrox::plasma_beta.avg", "7:grhydrox::plasma_beta.stddev", "8:grhydrox::plasma_beta.volume", "9:grhydrox::plasma_beta.L1norm", "10:grhydrox::plasma_beta.L2norm", "11:grhydrox::plasma_beta.maxabs", "12:grhydrox::plasma_beta.minloc[0]", "13:grhydrox::plasma_beta.minloc[1]", "14:grhydrox::plasma_beta.minloc[2]", "15:grhydrox::plasma_beta.maxloc[0]", "16:grhydrox::plasma_beta.maxloc[1]", "17:grhydrox::plasma_beta.maxloc[2]"]

#------------------------------------------------------------------------------ 
#Determine the latest output number that is already present in out_path (files before that will not be copied)
var_name = "admbase-lapse"
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
    plt.xlim(-0.4, 160)
    #plt.xlim(80, 150)
else:
    time_factor = 1.0 
    offset = 0.0
    plt.xlim(71800, 73500)
#------------------------------------------------------------------------------ 

plt.title("{}".format(var_name))
#plt.ylim(0.0004, 0.00063)


ring_marker_size = 40
dot_marker_size = 20

style = False #Always start with false so that dot is plotted first, especially when separate_chkpts=False
for output_number in range(0, latest_copied_output+1):     
    file_name = "norms/{}-output-{}.tsv".format(var_name, str(output_number).zfill(4))
    data = pd.read_csv(file_name, sep='\t', names=hydro_vars, comment="#");
    if style:  
        plt.scatter(data["2:time"]/time_factor-offset, data["3:admbase::alp.min"], s=ring_marker_size, marker="o", facecolors='none', edgecolors='b')
    else:
        plt.scatter(data["2:time"]/time_factor-offset, data["3:admbase::alp.min"], s=dot_marker_size, marker=".", facecolors='r', edgecolors='r')
    #Only toggle styles if we want to separate checkpoints
    if separate_chkpts:
        style = not style
plt.savefig("alp.png")
plt.show()
plt.close()

#----------------------------------------------------------------------------------------------------
plt.title("PNS position (alp) vs time")
plt.xlabel("time (ms)")
plt.ylabel("x-position of PNS (M)")
plt.xlim(-0.4, 160)

style = False #Always start with false so that dot is plotted first, especially when separate_chkpts=False
for output_number in range(0, latest_copied_output+1):     
    file_name = "norms/{}-output-{}.tsv".format(var_name, str(output_number).zfill(4))
    data = pd.read_csv(file_name, sep='\t', names=hydro_vars, comment="#");
    if style:  
        plt.scatter(data["2:time"]/time_factor-offset, data["12:grhydrox::plasma_beta.minloc[0]"], s=ring_marker_size, marker="o", facecolors='none', edgecolors='b')
    else:
        plt.scatter(data["2:time"]/time_factor-offset, data["12:grhydrox::plasma_beta.minloc[0]"], s=dot_marker_size, marker=".", facecolors='r', edgecolors='r')
    #Only toggle styles if we want to separate checkpoints
    if separate_chkpts:
        style = not style

plt.savefig("PNS_x_position_rho.png")    
plt.show()                




