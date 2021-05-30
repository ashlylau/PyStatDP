#!/bin/bash

# Set default parameters
privacy=1.0
e_iter=100
d_iter=500
pred_len=3
n_checks=3
test_range=0.5
train_privacy=1.0
algorithm=iris
impl=ibm
sampling_type=over
sampling_ratio=0.5

for arg in "$@"
do
    case $arg in
        -privacy=*) privacy="${arg#*=}";;
        -e_iter=*) e_iter="${arg#*=}";;
        -d_iter=*) d_iter="${arg#*=}";;
        -pred_len=*) pred_len="${arg#*=}";;
        -n_checks=*) n_checks="${arg#*=}";;
        -test_range=*) test_range="${arg#*=}";;
        -train_privacy=*) train_privacy="${arg#*=}";;
        -algorithm=*) algorithm="${arg#*=}";;
        -impl=*) impl="${arg#*=}";;
        -sampling_type=*) sampling_type="${arg#*=}";;
        -sampling_ratio=*) sampling_ratio="${arg#*=}";;
        -misclassified) misclassified=--misclassified;;
        -misclassified_minority) misclassified_minority=--misclassified_minority;;
        -full_dataset) full_dataset=--full_dataset;;
    esac
done

python --version
source ~/.bashrc
python --version
TERM=vt100 # or TERM=xterm
python3 -u -W ignore iris.py --privacy $privacy --e_iter $e_iter --d_iter $d_iter --pred_len $pred_len --n_checks $n_checks --test_range $test_range --train_privacy $train_privacy --algorithm $algorithm --impl $impl --sampling_ratio $sampling_ratio --sampling_type $sampling_type $misclassified $misclassified_minority $full_dataset 
uptime
