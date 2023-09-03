RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color
echo -e "${GREEN}Copying plot.py to MMAAMS..${NC}"
scp plot.py sshanka@u035042.science.uva.nl:/home/sshanka/CCSN_12000km_analysis/
echo " "
echo -e "${GREEN}Running plot.py on MMAAMS..${NC}"
ssh sshanka@u035042.science.uva.nl < commands.txt
echo " "
echo -e "${GREEN}Copying png files to local machine..${NC}"
scp sshanka@u035042.science.uva.nl:/home/sshanka/CCSN_12000km_analysis/*.png /home/swapnnil/Desktop/000_carpetX/NERSC_GPU_Hackathon/Balsara_files/test-GRHydroX-GPU-1/CCSN_12000km_analysis/openpmd/MMAAMS/
