#!/bin/bash 
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # GPU  
#SBATCH --mem=100G               # total memory
#SBATCH --time=15:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email
#SBATCH --mail-user=cw55@princeton.edu


# load modules or conda environments here
source /usr/share/Modules/init/bash
module purge 
# User specific environment and startup programs
module load anaconda3/2022.5
source /usr/licensed/anaconda3/2022.5/etc/profile.d/conda.sh
conda activate pytorch20

python -u sw_Li5Relu.py --sky_cond=cs  --eng_loss=Y --ensemble_size=2
python -u sw_Li5Relu.py --sky_cond=all --eng_loss=Y --ensemble_size=2
python -u sw_Li5Relu.py --sky_cond=cs  --eng_loss=N --ensemble_size=2
python -u sw_Li5Relu.py --sky_cond=all --eng_loss=N --ensemble_size=2





