#!/bin/bash 
#$ -o astdout_1000_16_rec.txt
#$ -e astderr_1000_16_rec.txt
#$ -j y 
#$ -N ClCH4rec16
#$ -l h_rt=9996:00:00 
#$ -l mem_total=1G
#$ -l long 
#$ -pe singlenode 24
cd 
source .bash_profile 
source /opt/intel/Compiler/11.0/074/bin/intel64/ifortvars_intel64.sh 
cd /home/ysuleyma/RPMDrate/ 
python rpmdrate.py /home/ysuleyma/RPMDrate/Examples/Cl+CH4/input_rec.py 1000 16 -p 24
 
