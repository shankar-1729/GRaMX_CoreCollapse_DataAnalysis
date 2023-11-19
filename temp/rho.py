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


hydro_vars = ["1:iteration", "2:time", "3:hydrobase::rho.min", "4:hydrobase::rho.max", "5:hydrobase::rho.sum", "6:hydrobase::rho.avg", "7:hydrobase::rho.stddev", "8:hydrobase::rho.volume", "9:hydrobase::rho.L1norm", "10:hydrobase::rho.L2norm", "11:hydrobase::rho.maxabs", "12:hydrobase::rho.minloc[0]", "13:hydrobase::rho.minloc[1]", "14:hydrobase::rho.minloc[2]", "15:hydrobase::rho.maxloc[0]", "16:hydrobase::rho.maxloc[1]", "17:hydrobase::rho.maxloc[2]"]

#------------------------------------------------------------------------------ 
#Determine the latest output number that is already present in out_path (files before that will not be copied)
var_name = "hydrobase-rho"
os.system("ls ../norms/ | grep {} > temp2.txt".format(var_name))
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
plt.ylim(0.0005, 0.001)

rho_noLeak_central_code = 0.0005276
rho_Leak_peak_code = 0.000607
plt.axhline(y = rho_noLeak_central_code, color = 'black', linestyle = '--')
plt.axhline(y = rho_Leak_peak_code, color = 'black', linestyle = '--')

ring_marker_size = 40
dot_marker_size = 20

style = False #Always start with false so that dot is plotted first, especially when separate_chkpts=False
for output_number in range(0, latest_copied_output+1):     
    file_name = "../norms/{}-output-{}.tsv".format(var_name, str(output_number).zfill(4))
    data = pd.read_csv(file_name, sep='\t', names=hydro_vars, comment="#");
    if style:  
        plt.scatter(data["2:time"]/time_factor-offset, data["4:hydrobase::rho.max"], s=ring_marker_size, marker="o", facecolors='none', edgecolors='b')
    else:
        plt.scatter(data["2:time"]/time_factor-offset, data["4:hydrobase::rho.max"], s=dot_marker_size, marker=".", facecolors='r', edgecolors='r')
    #Only toggle styles if we want to separate checkpoints
    if separate_chkpts:
        style = not style



style = False #Always start with false so that dot is plotted first, especially when separate_chkpts=False
for output_number in range(2, 12):     
    file_name = "temp{}/{}.tsv".format(output_number, var_name)
    data = pd.read_csv(file_name, sep='\t', names=hydro_vars, comment="#");
    if style:  
        plt.scatter(data["2:time"]/time_factor-offset, data["4:hydrobase::rho.max"], s=ring_marker_size, marker="o", facecolors='none', edgecolors='g')
    else:
        plt.scatter(data["2:time"]/time_factor-offset, data["4:hydrobase::rho.max"], s=dot_marker_size, marker=".", facecolors='m', edgecolors='m')
    #Only toggle styles if we want to separate checkpoints
    if separate_chkpts:
        style = not style


plt.savefig("rho.png")    
#plt.show()
plt.close()

#----------------------------------------------------------------------------------------------------

plt.title("PNS position (rho) vs time: x=blue, y=red, z=green")
plt.xlabel("time (ms)")
plt.ylabel("x,y,z-position of PNS (M)")
plt.xlim(-0.4, 160)

fit_file0 = open('PNS_position_vs_time_original.txt', 'w')
fit_file = open('PNS_position_vs_time.txt', 'w')


for output_number in range(0, latest_copied_output+1):     
    file_name = "../norms/{}-output-{}.tsv".format(var_name, str(output_number).zfill(4))
    data = pd.read_csv(file_name, sep='\t', names=hydro_vars, comment="#");   
    plt.scatter(data["2:time"]/time_factor-offset, data["15:hydrobase::rho.maxloc[0]"], s=10, marker="o", facecolors='none', edgecolors='b')
    plt.scatter(data["2:time"]/time_factor-offset, data["16:hydrobase::rho.maxloc[1]"], s=10, marker="*", facecolors='none', edgecolors='r')
    plt.scatter(data["2:time"]/time_factor-offset, data["17:hydrobase::rho.maxloc[2]"], s=10, marker="^", facecolors='none', edgecolors='g')
    
    if (output_number >= 104):
        #write data to file for fitting original
        print("first valid index = {}, last valid index = {}".format(data.first_valid_index(), data.last_valid_index()))
        for i in range(data.first_valid_index(), data.last_valid_index()):
            fit_file0.write("{}  {}\n".format(data["2:time"][i]/time_factor-offset, data["15:hydrobase::rho.maxloc[0]"][i]))
    
    if (output_number >= 104 and output_number <= 108):
        #write data to file for fitting modified
        print("first valid index = {}, last valid index = {}".format(data.first_valid_index(), data.last_valid_index()))
        for i in range(data.first_valid_index(), data.last_valid_index()):
            fit_file.write("{}  {}\n".format(data["2:time"][i]/time_factor-offset, data["15:hydrobase::rho.maxloc[0]"][i]))


for output_number in range(2, 13):     
    file_name = "temp{}/{}.tsv".format(output_number, var_name)
    data = pd.read_csv(file_name, sep='\t', names=hydro_vars, comment="#");   
    plt.scatter(data["2:time"]/time_factor-offset, data["15:hydrobase::rho.maxloc[0]"], s=10, marker="o", facecolors='none', edgecolors='m')
    plt.scatter(data["2:time"]/time_factor-offset, data["16:hydrobase::rho.maxloc[1]"], s=10, marker="*", facecolors='none', edgecolors='brown')
    plt.scatter(data["2:time"]/time_factor-offset, data["17:hydrobase::rho.maxloc[2]"], s=10, marker="^", facecolors='none', edgecolors='purple')
    
    #Write data to file for fitting
    print("first valid index = {}, last valid index = {}".format(data.first_valid_index(), data.last_valid_index()))
    for i in range(data.first_valid_index(), data.last_valid_index()):
        fit_file.write("{}  {}\n".format(data["2:time"][i]/time_factor-offset, data["15:hydrobase::rho.maxloc[0]"][i]))

        
plt.savefig("PNS_xyz_position_rho.png")    
plt.show()        

'''
for output_number in range(1, latest_copied_output+1):     
    file_name = "../norms/{}-output-{}.tsv".format(var_name, str(output_number).zfill(4))
    data = pd.read_csv(file_name, sep='\t', names=hydro_vars, comment="#");
    time_start = round(data["2:time"][0], 2)
    time_end = round(data["2:time"][data.last_valid_index()], 2)
    time_elapsed = time_end - time_start
    print("Output-{}({}-{});    Time elapsed: {} M, {} ms;      Rate = {} M/hr, {} ms/hr".format(str(output_number).zfill(4), round(time_start/203.0-offset,2), round(time_end/203.0-offset,2), round(time_elapsed,2), round(time_elapsed/203.0, 2), round(time_elapsed/2,2), round(time_elapsed/203.0/2.0,2)))
'''




