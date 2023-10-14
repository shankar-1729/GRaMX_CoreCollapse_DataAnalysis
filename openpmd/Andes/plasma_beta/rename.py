import os


os.system("ls *.png > temp.txt")

file1 = open('temp.txt', 'r')
Lines = file1.readlines()
for line in Lines:
    tempvar = line.strip().split('_')[2].split("it")[1].zfill(6)
    out_name = "plasma_beta_it{}_yz.png".format(tempvar) 
    #print(out_name)
    command = "mv {} {}".format(line.strip(), out_name)
    print(command)
    os.system(command)
