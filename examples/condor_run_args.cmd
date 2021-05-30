Executable = /homes/al5217/PyStatDP/examples/iris_condor_args.sh
Universe = vanilla
Output = /homes/al5217/PyStatDP/examples/job-$(Cluster)-$(Process).out
Error = /homes/al5217/PyStatDP/examples/job-$(Cluster)-$(Process).out
Requirements = regexp("^(corona|edge|line|sprite|ray|texel|curve)[0-9][0-9]", TARGET.machine)


arguments = -privacy=1.5 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=100 -algorithm=machine -impl=ibm-linreg
Queue

arguments = -privacy=1.5 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150 -algorithm=machine -impl=ibm-linreg
Queue

arguments = -privacy=1.5 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=200 -algorithm=machine -impl=ibm-linreg
Queue

arguments = -privacy=1.5 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=250 -algorithm=machine -impl=ibm-linreg
Queue

arguments = -privacy=1.5 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=500 -algorithm=machine -impl=ibm-linreg
Queue


