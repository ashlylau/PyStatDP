#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=al5217 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/al5217/env/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
python3 -u -W ignore iris.py --privacy 0.15001 --e_iter 1000 --d_iter 5000 --pred_len 5 --n_checks 9 --test_range 0.05 --train_privacy 0.2 --algorithm iris --misclassified --full_dataset --useIbmNB
uptime
