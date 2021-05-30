Executable = /homes/al5217/PyStatDP/examples/iris_condor_args.sh
Universe = vanilla
Output = /homes/al5217/PyStatDP/examples/job-$(Cluster)-$(Process).out
Error = /homes/al5217/PyStatDP/examples/job-$(Cluster)-$(Process).out
Requirements = regexp("^(corona|edge|line|sprite|ray|texel|curve)[0-9][0-9]", TARGET.machine)



arguments = -privacy=1.2 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.2
Queue

arguments = -privacy=1.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.3
Queue

arguments = -privacy=2.2 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.4
Queue

arguments = -privacy=2.2 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.5
Queue

arguments = -privacy=1.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.6
Queue

arguments = -privacy=2.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.7
Queue

arguments = -privacy=3.2 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.8
Queue

arguments = -privacy=3.2 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.9
Queue

arguments = -privacy=2.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=1.0
Queue





arguments = -privacy=2.2 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=under -sampling_ratio=0.2
Queue

arguments = -privacy=1.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=under -sampling_ratio=0.3
Queue

arguments = -privacy=1.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=under -sampling_ratio=0.4
Queue

arguments = -privacy=2.2 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=under -sampling_ratio=0.5
Queue

arguments = -privacy=1.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=under -sampling_ratio=0.6
Queue

arguments = -privacy=1.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=under -sampling_ratio=0.7
Queue

arguments = -privacy=2.2 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=under -sampling_ratio=0.8
Queue

arguments = -privacy=1.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=under -sampling_ratio=0.9
Queue

arguments = -privacy=1.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=under -sampling_ratio=1.0
Queue




arguments = -privacy=1.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=smote -sampling_ratio=0.2
Queue

arguments = -privacy=1.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=smote -sampling_ratio=0.3
Queue

arguments = -privacy=2.2 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=smote -sampling_ratio=0.4
Queue

arguments = -privacy=1.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=smote -sampling_ratio=0.5
Queue

arguments = -privacy=2.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=smote -sampling_ratio=0.6
Queue

arguments = -privacy=2.2 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=smote -sampling_ratio=0.7
Queue

arguments = -privacy=2.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=smote -sampling_ratio=0.8
Queue

arguments = -privacy=1.7 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=smote -sampling_ratio=0.9
Queue

arguments = -privacy=3.1 -e_iter=500 -d_iter=2500 -pred_len=2 -n_checks=9 -test_range=0.1 -train_privacy=150.0 -algorithm=machine-resample -impl=ibm-linreg -sampling_type=smote -sampling_ratio=1.0
Queue
