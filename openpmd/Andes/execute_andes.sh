RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color
echo -e "${GREEN}Copying plot.py to Andes..${NC}"
scp -oControlMaster=auto -oControlPath=~/.ssh/copy_code-%C -oControlPersist=3600 ../plot.py  sshanka@andes.olcf.ornl.gov:/gpfs/alpine/ast154/scratch/sshanka/carpetx_github/openpmd/
echo " "
echo -e "${GREEN}Running plot.py on Andes..${NC}"
ssh -oControlMaster=auto -oControlPath=~/.ssh/plot_code-%C -oControlPersist=3600 sshanka@andes.olcf.ornl.gov < commands_andes.txt
echo " "

echo "sleeping for 5 seconds"
sleep 5s
echo -e "${GREEN}Copying png files to local machine..${NC}"
rsync -e "ssh -oControlMaster=auto -oControlPath=~/.ssh/plot_code-%C -oControlPersist=3600" -rtv sshanka@andes.olcf.ornl.gov:/gpfs/alpine/ast154/scratch/sshanka/carpetx_github/openpmd/*.png  /home/swapnnil/Desktop/000_carpetX/NERSC_GPU_Hackathon/Balsara_files/test-GRHydroX-GPU-1/CCSN_12000km_analysis/openpmd/Andes/

