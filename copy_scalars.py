import os

#global variables
out_path = "/gpfs/alpine/ast154/scratch/sshanka/carpetx_github/CCSN_12000km_analysis/scalars" 
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
        cp_source = "{}/output-{}/{}/".format(sim_path, str(i).zfill(4), parfile_name)
        print("Copying scalars from {}...".format(cp_source))
        os.system("ls {} | grep {}.it > temp3.txt".format(cp_source, var_name))
        
        output_file = open("scalars/{}-output-{}.tsv".format(var_name, str(i).zfill(4)), "w")
        
        header_flag = 0
        file3 = open('temp3.txt', 'r')
        Lines = file3.readlines()
        for line in Lines:
            #print(line)
            file4 = cp_source + line.strip()
            #print(file4)
            with open(file4, 'r') as fp:
                if header_flag == 0:
                    output_file.write(fp.readlines()[0])

            with open(file4, 'r') as fp:
                output_file.write(fp.readlines()[1])
                header_flag = header_flag + 1
                #print(x)

        file3.close()
        os.system("rm temp3.txt")
        output_file.close()
#-----------------------------------------------------------------------------------------------------------------------

var_names = ["shocktracker-max_shock_radius", "shocktracker-max_shock_radius_x", "shocktracker-max_shock_radius_y", "shocktracker-max_shock_radius_z"]

for var_name in var_names:
    execute_copy(var_name)
    print("")




