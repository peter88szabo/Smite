#!/bin/bash

#$ -o astdout_400_128.txt
#$ -e astderr_400_128.txt
#$ -j y
#$ -N OH+CH4400
#$ -l h_rt=9996:00:00
#$ -l mem_total=1G
#$ -l long 
#$ -pe singlenode 8

cd
source .bash_profile

source /opt/intel/Compiler/11.0/074/bin/intel64/ifortvars_intel64.sh

# Run RPMD
cd /home/ysuleyma/RPMDrate
python rpmdrate.py /home/ysuleyma/RPMDrate/Examples/OH+CH4/input.py 400 128 -p 8

