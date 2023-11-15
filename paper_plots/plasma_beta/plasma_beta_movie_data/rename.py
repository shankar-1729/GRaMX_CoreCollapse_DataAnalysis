import os


os.system("ls *.png > temp.txt")

file1 = open('temp.txt', 'r')
Lines = file1.readlines()
for line in Lines:
    print line
    tempvar = line.strip().split('_')[3].split("it")[1].split(".")[0].zfill(6)
    out_name = "plasma_beta_xz_it{}.png".format(tempvar) 
    #print(out_name)
    command = "mv {} {}".format(line.strip(), out_name)
    print(command)
    os.system(command)
    
