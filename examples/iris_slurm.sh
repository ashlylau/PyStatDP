#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=al5217 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/al5217/env/bin/:$PATH
source activate
source /vol/cuda/11.1.0-cudnn8.0.4.30/setup.sh
TERM=vt100 # or TERM=xterm
python3 -u -W ignore iris.py --privacy 0.4 --e_iter 500 --d_iter 5000 --pred_len 5 --n_checks 5 --test_range 0.2 --train_privacy 1.0 --algorithm iris --misclassified --full_dataset
/usr/bin/nvidia-smi
uptime
