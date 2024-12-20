#!/bin/bash 
#$ -o stdout_300_1.txt
#$ -e stderr_300_1.txt
#$ -j y 
#$ -N FHD 
#$ -l h_rt=9996:00:00 
#$ -l mem_total=1G
#$ -l long 
#$ -pe singlenode 1
cd 
source .bash_profile 
source /home/jwallen/virtualenv/bin/activate 
source /opt/intel/Compiler/11.0/074/bin/intel64/ifortvars_intel64.sh 
cd /home/ysuleyma/RPMDrate/ 
python rpmdrate.py /home/ysuleyma/RPMDrate/Examples/FH2N5Z/input.py 300 1 -p 1
 
# Turn off virtualenv when done 
deactivate 
