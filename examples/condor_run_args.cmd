Executable = /homes/al5217/PyStatDP/examples/iris_condor_args.sh
Universe = vanilla
Output = /homes/al5217/PyStatDP/examples/job-$(Cluster)-$(Process).out
Error = /homes/al5217/PyStatDP/examples/job-$(Cluster)-$(Process).out
Requirements = regexp("^(corona|edge|line|pixel|ray|texel|)[0-9][0-9]", TARGET.Machine)


arguments = -privacy=0.4 -e_iter=1000 -d_iter=5000 -pred_len=3 -n_checks=9 -test_range=0.1 -train_privacy=1.0 -algorithm=adult-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.33 -misclassified
Queue

arguments = -privacy=0.4 -e_iter=1000 -d_iter=5000 -pred_len=3 -n_checks=9 -test_range=0.1 -train_privacy=1.0 -algorithm=adult-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.4 -misclassified
Queue

arguments = -privacy=0.4 -e_iter=1000 -d_iter=5000 -pred_len=3 -n_checks=9 -test_range=0.1 -train_privacy=1.0 -algorithm=adult-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.6 -misclassified
Queue

arguments = -privacy=0.4 -e_iter=1000 -d_iter=5000 -pred_len=3 -n_checks=9 -test_range=0.1 -train_privacy=1.0 -algorithm=adult-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.8 -misclassified
Queue

arguments = -privacy=0.4 -e_iter=1000 -d_iter=5000 -pred_len=3 -n_checks=9 -test_range=0.1 -train_privacy=1.0 -algorithm=adult-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=1.0 -misclassified
Queue



arguments = -privacy=2.0 -e_iter=1000 -d_iter=5000 -pred_len=3 -n_checks=9 -test_range=0.5 -train_privacy=10.0 -algorithm=adult-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.33 -misclassified
Queue

arguments = -privacy=2.0 -e_iter=1000 -d_iter=5000 -pred_len=3 -n_checks=9 -test_range=0.5 -train_privacy=10.0 -algorithm=adult-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.4 -misclassified
Queue

arguments = -privacy=2.0 -e_iter=1000 -d_iter=5000 -pred_len=3 -n_checks=9 -test_range=0.5 -train_privacy=10.0 -algorithm=adult-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.6 -misclassified
Queue

arguments = -privacy=2.0 -e_iter=1000 -d_iter=5000 -pred_len=3 -n_checks=9 -test_range=0.5 -train_privacy=10.0 -algorithm=adult-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.8 -misclassified
Queue

arguments = -privacy=2.0 -e_iter=1000 -d_iter=5000 -pred_len=3 -n_checks=9 -test_range=0.5 -train_privacy=10.0 -algorithm=adult-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=1.0 -misclassified
Queue



arguments = -privacy=5.0 -e_iter=1000 -d_iter=5000 -pred_len=3 -n_checks=9 -test_range=1.0 -train_privacy=100.0 -algorithm=adult-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.33 -misclassified
Queue

arguments = -privacy=5.0 -e_iter=1000 -d_iter=5000 -pred_len=3 -n_checks=9 -test_range=1.0 -train_privacy=100.0 -algorithm=adult-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.4 -misclassified
Queue

arguments = -privacy=5.0 -e_iter=1000 -d_iter=5000 -pred_len=3 -n_checks=9 -test_range=1.0 -train_privacy=100.0 -algorithm=adult-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.6 -misclassified
Queue

arguments = -privacy=5.0 -e_iter=1000 -d_iter=5000 -pred_len=3 -n_checks=9 -test_range=1.0 -train_privacy=100.0 -algorithm=adult-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=0.8 -misclassified
Queue

arguments = -privacy=5.0 -e_iter=1000 -d_iter=5000 -pred_len=3 -n_checks=9 -test_range=1.0 -train_privacy=100.0 -algorithm=adult-resample -impl=ibm-linreg -sampling_type=over -sampling_ratio=1.0 -misclassified
Queue
