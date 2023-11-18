RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color
echo -e "${GREEN}Copying plot.py to Andes..${NC}"

scp -oControlMaster=auto -oControlPath=~/.ssh/copy_code-%C -oControlPersist=3600 ejecta_mass_energy.py  sshanka@andes.olcf.ornl.gov:/lustre/orion/ast154/scratch/sshanka/simulations/paper_plots/plot.py
#scp -oControlMaster=auto -oControlPath=~/.ssh/copy_code-%C -oControlPersist=3600 plot_final.py  sshanka@andes.olcf.ornl.gov:/lustre/orion/ast154/scratch/sshanka/simulations/paper_plots/plot.py

echo " "
#echo -e "${GREEN}Running plot.py on Andes..${NC}"
#ssh -oControlMaster=auto -oControlPath=~/.ssh/plot_code-%C -oControlPersist=3600 sshanka@andes.olcf.ornl.gov < commands_andes.txt
echo " "

#echo "sleeping for 5 seconds"
#sleep 5s
#echo -e "${GREEN}Copying png files to local machine..${NC}"
#rsync -e "ssh -oControlMaster=auto -oControlPath=~/.ssh/plot_code-%C -oControlPersist=3600" -rtv sshanka@andes.olcf.ornl.gov:/gpfs/alpine/ast154/scratch/sshanka/carpetx_github/openpmd/all_output/*.png  /home/swapnnil/Desktop/000_carpetX/NERSC_GPU_Hackathon/Balsara_files/test-GRHydroX-GPU-1/CCSN_12000km_analysis/openpmd/Andes/temp/

#rsync -e "ssh -oControlMaster=auto -oControlPath=~/.ssh/plot_code-%C -oControlPersist=3600" -rtv sshanka@andes.olcf.ornl.gov:/gpfs/alpine/ast154/scratch/sshanka/carpetx_github/openpmd/hydrobase_entropy/*.png  /home/swapnnil/Desktop/000_carpetX/NERSC_GPU_Hackathon/Balsara_files/test-GRHydroX-GPU-1/CCSN_12000km_analysis/openpmd/Andes/hydrobase_entropy/

#rsync -e "ssh -oControlMaster=auto -oControlPath=~/.ssh/plot_code-%C -oControlPersist=3600" -rtv sshanka@andes.olcf.ornl.gov:/gpfs/alpine/ast154/scratch/sshanka/carpetx_github/openpmd/plasma_beta/*.png  /home/swapnnil/Desktop/000_carpetX/NERSC_GPU_Hackathon/Balsara_files/test-GRHydroX-GPU-1/CCSN_12000km_analysis/openpmd/Andes/plasma_beta/

#rsync -e "ssh -oControlMaster=auto -oControlPath=~/.ssh/plot_code-%C -oControlPersist=3600" -rtv sshanka@andes.olcf.ornl.gov:/gpfs/alpine/ast154/scratch/sshanka/carpetx_github/openpmd/magnetisation/*.png  /home/swapnnil/Desktop/000_carpetX/NERSC_GPU_Hackathon/Balsara_files/test-GRHydroX-GPU-1/CCSN_12000km_analysis/openpmd/Andes/magnetisation/

#TODO: For hydrobase_ye
#rsync -e "ssh -oControlMaster=auto -oControlPath=~/.ssh/plot_code-%C -oControlPersist=3600" -rtv sshanka@andes.olcf.ornl.gov:/lustre/orion/ast154/scratch/sshanka/simulations/paper_plots/hydrobase_ye/*.png  /home/swapnnil/Desktop/000_carpetX/NERSC_GPU_Hackathon/Balsara_files/test-GRHydroX-GPU-1/CCSN_12000km_analysis/paper_plots/hydrobase_ye/


