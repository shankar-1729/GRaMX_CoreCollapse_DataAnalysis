import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


#To refresh data:
#$cd /home/swapnnil/Desktop/000_carpetX/NERSC_GPU_Hackathon/Balsara_files/test-GRHydroX-GPU-1/CCSN_12000km_analysis
#$rsync -rtv sshanka@andes.olcf.ornl.gov:/gpfs/alpine/ast154/scratch/sshanka/carpetx_github/CCSN_12000km_analysis/ .


hydro_vars = [ "1:iteration",	"2:time",	"3:neutrinoleakage::abs_number.min",	"4:neutrinoleakage::abs_number.max",	"5:neutrinoleakage::abs_number.sum",	"6:neutrinoleakage::abs_number.avg",	"7:neutrinoleakage::abs_number.stddev",	"8:neutrinoleakage::abs_number.volume",	"9:neutrinoleakage::abs_number.L1norm",	"10:neutrinoleakage::abs_number.L2norm",	"11:neutrinoleakage::abs_number.maxabs",	"12:neutrinoleakage::abs_number.minloc[0]",	"13:neutrinoleakage::abs_number.minloc[1]",	"14:neutrinoleakage::abs_number.minloc[2]",	"15:neutrinoleakage::abs_number.maxloc[0]",	"16:neutrinoleakage::abs_number.maxloc[1]",	"17:neutrinoleakage::abs_number.maxloc[2]",	"18:neutrinoleakage::abs_energy.min",	"19:neutrinoleakage::abs_energy.max",	"20:neutrinoleakage::abs_energy.sum",	"21:neutrinoleakage::abs_energy.avg",	"22:neutrinoleakage::abs_energy.stddev",	"23:neutrinoleakage::abs_energy.volume",	"24:neutrinoleakage::abs_energy.L1norm",	"25:neutrinoleakage::abs_energy.L2norm",	"26:neutrinoleakage::abs_energy.maxabs",	"27:neutrinoleakage::abs_energy.minloc[0]",	"28:neutrinoleakage::abs_energy.minloc[1]",	"29:neutrinoleakage::abs_energy.minloc[2]",	"30:neutrinoleakage::abs_energy.maxloc[0]",	"31:neutrinoleakage::abs_energy.maxloc[1]",	"32:neutrinoleakage::abs_energy.maxloc[2]"]

#------------------------------------------------------------------------------ 
#Determine the latest output number that is already present in out_path (files before that will not be copied)
var_name = "neutrinoleakage-neutrinoleakage_abs"
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
separate_chkpts = True
display_physical_time = True

if display_physical_time:
    time_factor = 203.0
    offset = 71928/203.0
    #plt.xlim(-0.4, 150)
    plt.xlim(80, 160)
else:
    time_factor = 1.0 
    offset = 0.0
    plt.xlim(71800, 73500)
#------------------------------------------------------------------------------    


plt.title("{}::abs_number".format(var_name))
plt.ylim(-2e-6, 0.000084)

ring_marker_size = 20 #40
dot_marker_size = 20

style = False #Always start with false so that dot is plotted first, especially when separate_chkpts=False
for output_number in range(0, latest_copied_output+1):     
    file_name = "norms/{}-output-{}.tsv".format(var_name, str(output_number).zfill(4))
    data = pd.read_csv(file_name, sep='\t', names=hydro_vars, comment="#");
    if style:  
        plt.scatter(data["2:time"]/time_factor-offset, data["5:neutrinoleakage::abs_number.sum"], s=ring_marker_size, marker="o", facecolors='none', edgecolors='b')
    else:
        plt.scatter(data["2:time"]/time_factor-offset, data["5:neutrinoleakage::abs_number.sum"], s=dot_marker_size, marker=".", facecolors='r', edgecolors='r')
    #Only toggle styles if we want to separate checkpoints
    if separate_chkpts:
        style = not style

    
plt.show()


