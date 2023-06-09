#!/bin/bash
#SBATCH --nodes=1                 # node count
#SBATCH --sockets-per-node=1
#SBATCH  --cores-per-socket=10
#SBATCH --mem=380G
#SBATCH -t 06:30:00
# Sends mail when process begins, and when it ends. 
# Make sure you define your email
#SBATCH --mail-type=all
#SBATCH --mail-user=cw55@princeton.edu

###################################################################
#Script Name : run_notebook_analysis
#Description : run and save the notebook via slurm
#Args        :
#Author      : Chenggong Wang
#Email       : c.wang@princeton.edu
###################################################################

set -e #end with any error
#set -x #expands variables and prints a little + sign before the line
# load modules or conda environments here
source /usr/share/Modules/init/bash

module purge
module load anaconda3/2023.3

source /usr/licensed/anaconda3/2023.3/etc/profile.d/conda.sh
conda activate pytorch20
which python
date_v=$(date +%y%m%d%H%M)

#exec=std_lw_analysis_regrid_map_train
#echo $(date +%y%m%d%H%M)
#jupyter nbconvert --to notebook --execute $exec --output $exec.$date_v
#echo $(date +%y%m%d%H%M)
#exec=std_lw_analysis_regrid_map_test
#echo $(date +%y%m%d%H%M)
#jupyter nbconvert --to notebook --execute $exec --output $exec.$date_v
#echo $(date +%y%m%d%H%M)

exec=std_sw_analysis_regrid_map_train
echo $(date +%y%m%d%H%M)
jupyter nbconvert --to notebook --execute $exec --output $exec.$date_v
echo $(date +%y%m%d%H%M)
exec=std_sw_analysis_regrid_map_test
echo $(date +%y%m%d%H%M)
jupyter nbconvert --to notebook --execute $exec --output $exec.$date_v
echo $(date +%y%m%d%H%M)

echo END
