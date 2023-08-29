import os

#global variables
out_path = "/gpfs/alpine/ast154/scratch/sshanka/carpetx_github/CCSN_12000km_analysis/norms" 
parfile_name = "CCSN_12000km"
sim_path = "/gpfs/alpine/ast154/scratch/sshanka/simulations/CCSN_12000km" 

from termcolor import colored, cprint
#-----------------------------------------------------------------------------------------------------------------
def execute_copy(var_name):
    colortext = colored("Copying files for variable {}...".format(var_name), "green")
    print(colortext)

    #Determine the currently active output number
    os.system("ls {} | grep active > temp1.txt".format(sim_path))
    file1 = open('temp1.txt', 'r')
    Lines = file1.readlines()
    current_output = int(Lines[0].strip().split('-')[1])
    print("current output number = {}".format(current_output))
    file1.close()
    os.system("rm temp1.txt")

    #Determine the latest output number that is already present in out_path (files before that will not be copied)
    os.system("ls {} | grep {} > temp2.txt".format(out_path, var_name))
    file1 = open('temp2.txt', 'r')
    Lines = file1.readlines()
    out_list = []
    for line in Lines:
        tempvar = int(line.strip().split('-')[3].split('.')[0])
        out_list.append(tempvar)
    
    latest_copied_output = 0
    if(len(out_list) > 0):
        latest_copied_output = max(out_list)
    
    print("latest copied output number = {}".format(latest_copied_output))
    file1.close()
    os.system("rm temp2.txt")

    start_out = latest_copied_output
    end_out = current_output

    for i in range(start_out, end_out+1):
        cp_source = "{}/output-{}/{}/norms/{}.tsv".format(sim_path, str(i).zfill(4), parfile_name, var_name)
        cp_dest = "{}/{}-output-{}.tsv".format(out_path, var_name, str(i).zfill(4))
        cp_command = "cp {} {}".format(cp_source, cp_dest)
        print(cp_command)
        os.system(cp_command)
#-----------------------------------------------------------------------------------------------------------------------

var_names = ["hydrobase-rho", "admbase-lapse", "hydrobase-entropy", "hydrobase-temperature", "z4c-allc", "hydrobase-bvec", "neutrinoleakage-neutrinoleakage_abs"]

for var_name in var_names:
    execute_copy(var_name)
    print("")




