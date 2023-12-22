import os

#global variables
out_path = "/lustre/orion/ast154/scratch/sshanka/simulations/paper_plots/norms" 
parfile_name = "CCSN_12000km"
sim_path = "/lustre/orion/ast191/scratch/sshanka/simulations/Ref6_40" 

#from termcolor import colored, cprint
#-----------------------------------------------------------------------------------------------------------------
def execute_copy(var_name):
    #colortext = colored("Copying files for variable {}...".format(var_name), "green")
    #print(colortext)

    valid_output_list = [18, 24, 25, 26, 30, 56, 57, 58, 59, 69, 70, 71, 72, 73, 74, 75]
    
    for i in valid_output_list:
        cp_source = "{}/output-{}/{}/norms/{}.tsv".format(sim_path, str(i).zfill(4), parfile_name, var_name)
        cp_dest = "{}/{}-output-{}.tsv".format(out_path, var_name, str(i).zfill(4))
        cp_command = "cp {} {}".format(cp_source, cp_dest)
        print(cp_command)
        os.system(cp_command)
        
    for i in range(76, 104):
        cp_source = "{}/output-{}/{}/norms/{}.tsv".format(sim_path, str(i).zfill(4), parfile_name, var_name)
        cp_dest = "{}/{}-output-{}.tsv".format(out_path, var_name, str(i).zfill(4))
        cp_command = "cp {} {}".format(cp_source, cp_dest)
        print(cp_command)
        os.system(cp_command)
#-----------------------------------------------------------------------------------------------------------------------

var_names = ["hydrobase-rho", "admbase-lapse", "hydrobase-entropy", "hydrobase-temperature", "z4c-allc", "hydrobase-bvec", "neutrinoleakage-neutrinoleakage_abs", "grhydrox-magnetization", "grhydrox-plasma_beta"]

for var_name in var_names:
    execute_copy(var_name)
    print("")




