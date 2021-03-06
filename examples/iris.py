
import argparse
import json
import os
import sys
import time

import diffprivlib.models as dp
import numpy as np
import torch

sys.path.append(os.path.abspath('../'))
from pathlib import Path

from diffprivlib.models import GaussianNB as IBMGaussianNB
from diffprivlib.models import LinearRegression as IBMLinearRegression
from diffprivlib.models import LogisticRegression as IBMLogisticRegression
from jsonpickle import encode
from pydp.ml.naive_bayes import GaussianNB as PyDPGaussianNB
from pystatdp import pystatdp
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as SKLGaussianNB
from sklearn.linear_model import LogisticRegression as SKLLogisticRegression
from sklearn.linear_model import LinearRegression as SKLLinearRegression
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


class Iris:
    def __init__(self, privacy, lower, upper, priors, probability, var_smoothing, X_train, y_train, X_test, pred_len, train_privacy, impl):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.pred_len = pred_len
        if impl == 'ibm':
            clf = IBMGaussianNB(epsilon=train_privacy, bounds=(lower, upper), var_smoothing=var_smoothing)
        elif impl == 'skl':
            clf = SKLGaussianNB(var_smoothing=var_smoothing)
        else:
            clf = PyDPGaussianNB(epsilon=train_privacy, bounds=(lower, upper), probability=probability, var_smoothing=var_smoothing)
        self.clf = clf

    def quick_result(self, data):
        i = data[0]
        if i != -1:
            self.X_train = np.append(self.X_train[:i-1], self.X_train[i:], axis=0)
            self.y_train = np.append(self.y_train[:i-1], self.y_train[i:], axis=0)
        self.clf.fit(self.X_train, self.y_train)
        return self.clf.predict(self.X_test[:self.pred_len]).tolist()


class Adult:
    def __init__(self, privacy, lower, upper, X_train, y_train, X_test, pred_len, train_privacy, impl='ibm-linreg'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.pred_len = pred_len
        self.impl = impl
        if impl == 'ibm-logreg':
            clf = IBMLogisticRegression(epsilon=train_privacy, data_norm=100)
        elif impl == 'skl-logreg':
            clf = SKLLogisticRegression(solver="lbfgs")
        elif impl == 'ibm-linreg':
            clf = IBMLinearRegression(epsilon=train_privacy, bounds_X=(lower, upper), bounds_y=(0.0,1.0))
        else: # If we want to explore GaussianNB for Adult.
            clf = PyDPGaussianNB(epsilon=train_privacy, bounds=(lower, upper))
        self.clf = clf

    def quick_result(self, data):
        i = data[0]
        j = 1000
        if i != -1:
            self.X_train = np.append(self.X_train[:i*j], self.X_train[i*j+j:], axis=0)
            self.y_train = np.append(self.y_train[:i*j], self.y_train[i*j+j:], axis=0)
        self.clf.fit(self.X_train, self.y_train)
        if self.impl in ['ibm-logreg', 'skl-logreg', 'pydp-nb']:
            # Return probabilities of minority class
            return self.clf.predict_proba(self.X_test[:self.pred_len])[:, 1].tolist()
        return self.clf.predict(self.X_test[:self.pred_len]).tolist()


class AdultResample(Adult):
    def __init__(self, privacy, sampling_type, sampling_ratio, *args):
        self.sampling_type = sampling_type
        self.sampling_ratio = sampling_ratio
        super().__init__(privacy, *args)

    def quick_result(self, data):
        i = data[0]
        j = 1000
        if i != -1:
            self.X_train = np.append(self.X_train[:i*j], self.X_train[i*j+j:], axis=0)
            self.y_train = np.append(self.y_train[:i*j], self.y_train[i*j+j:], axis=0)
        if self.sampling_type == 'over':
            self.X_train, self.y_train = RandomOverSampler(sampling_strategy=self.sampling_ratio).fit_resample(self.X_train, self.y_train)
        else:
            self.X_train, self.y_train = RandomUnderSampler(sampling_strategy=self.sampling_ratio).fit_resample(self.X_train, self.y_train)
            pass
        self.clf.fit(self.X_train, self.y_train)
        # Maybe here if we want to output categorical for linreg, use self.impl to find implementation and use threshold to change to categorical
        return self.clf.predict(self.X_test[:self.pred_len]).tolist()


class Diabetes:
    def __init__(self, privacy, bounds_X, bounds_y, X_train, y_train, X_test, pred_len, train_privacy):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.pred_len = pred_len
        regr = IBMLinearRegression(epsilon=train_privacy, bounds_X=bounds_X, bounds_y=bounds_y)
        self.regr = regr

    def quick_result(self, data):
        i = data[0]
        j = 10
        if i != -1:
            self.X_train = np.append(self.X_train[:i*j], self.X_train[i*j+j:], axis=0)
            self.y_train = np.append(self.y_train[:i*j], self.y_train[i*j+j:], axis=0)
        self.regr.fit(self.X_train, self.y_train)
        return self.regr.predict(self.X_test[:self.pred_len]).tolist()


class Pima:
    def __init__(self, privacy, lower, upper, X_train, y_train, X_test, pred_len, train_privacy, impl='ibm-nb'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.pred_len = pred_len
        self.impl = impl
        if impl == 'pydp-nb':
            clf = PyDPGaussianNB(epsilon=train_privacy, bounds=(lower, upper))
        elif impl == 'ibm-nb':
            clf = IBMGaussianNB(epsilon=train_privacy, bounds=(lower, upper))
        elif impl == 'ibm-logreg':
            clf = IBMLogisticRegression(epsilon=train_privacy, max_iter=10000, data_norm=10000)
        elif impl == 'ibm-linreg':
            clf = IBMLinearRegression(epsilon=train_privacy, bounds_X=(lower, upper), bounds_y=(0.0,1.0))
        else:
            clf = SKLGaussianNB()
        self.clf = clf

    def quick_result(self, data):
        i = data[0]
        j = 10
        if i != -1:
            self.X_train = np.append(self.X_train[:i*j], self.X_train[i*j+j:], axis=0)
            self.y_train = np.append(self.y_train[:i*j], self.y_train[i*j+j:], axis=0)
        self.clf.fit(self.X_train, self.y_train)
        return self.clf.predict(self.X_test[:self.pred_len]).tolist()


class Machine:
    def __init__(self, privacy, lower, upper, X_train, y_train, X_test, pred_len, train_privacy, impl='ibm-nb'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.pred_len = pred_len
        self.impl = impl
        if impl == 'ibm-linreg':
            clf = IBMLinearRegression(epsilon=train_privacy, bounds_X=(lower, upper), bounds_y=(0.0,1.0))
        else:
            clf = SKLLinearRegression()
        self.clf = clf

    def quick_result(self, data):
        i = data[0]
        j = 5
        if i != -1:
            self.X_train = np.append(self.X_train[:i*j], self.X_train[i*j+j:], axis=0)
            self.y_train = np.append(self.y_train[:i*j], self.y_train[i*j+j:], axis=0)
        self.clf.fit(self.X_train, self.y_train)
        return self.clf.predict(self.X_test[:self.pred_len]).tolist()

class MachineResample(Machine):
    def __init__(self, privacy, sampling_type, sampling_ratio, *args):
        self.sampling_type = sampling_type
        self.sampling_ratio = sampling_ratio
        super().__init__(privacy, *args)

    def quick_result(self, data):
        i = data[0]
        j = 5
        if i != -1:
            self.X_train = np.append(self.X_train[:i*j], self.X_train[i*j+j:], axis=0)
            self.y_train = np.append(self.y_train[:i*j], self.y_train[i*j+j:], axis=0)
        
        full_data = np.append(self.X_train, self.y_train.reshape(-1,1), axis=1)
        labels = np.where(self.y_train > 200, 1, 0)
        
        if self.sampling_type == 'over':
            full_data_over, _ = RandomOverSampler(sampling_strategy=self.sampling_ratio).fit_resample(full_data, labels)
            self.X_train = full_data_over[:,:-1]
            self.y_train = full_data_over[:,-1:].ravel()
        elif self.sampling_type == 'under':
            full_data_over, _ = RandomUnderSampler(sampling_strategy=self.sampling_ratio).fit_resample(full_data, labels)
            self.X_train = full_data_over[:,:-1]
            self.y_train = full_data_over[:,-1:].ravel()
        else: # SMOTE resampling
            full_data_over, _ = SMOTE(sampling_strategy=self.sampling_ratio).fit_resample(full_data, labels)
            self.X_train = full_data_over[:,:-1]
            self.y_train = full_data_over[:,-1:].ravel()

        self.clf.fit(self.X_train, self.y_train)
        return self.clf.predict(self.X_test[:self.pred_len]).tolist()


class Dummy:
    def __init__(self, privacy, pred_len):
        self.pred_len = pred_len

    def quick_result(self, data):
        return [0]*self.pred_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment outlier D and D\' models')
    parser.add_argument('--algorithm', default='iris', help='algorithm to test')
    parser.add_argument('--privacy', type=float, default=3.0, help='midpoint privacy budget for test range')
    parser.add_argument('--train_privacy', type=float, default=0.9, help='claimed privacy budget')
    parser.add_argument('--test_range', type=float, default=0.5, help='test range')
    parser.add_argument('--n_checks', type=int, default=3, help='number of tests to run')
    parser.add_argument('--threshold', type=float, default=0.05, help='threshold to determine measured epsilon -- alpha')
    parser.add_argument('--e_iter', type=int, default=100, help='number of iterations to run event selection')
    parser.add_argument('--d_iter', type=int, default=50, help='number of iterations to run detection algorithm')
    parser.add_argument('--pred_len', type=int, default=3, help='length of prediction output')
    parser.add_argument('--misclassified', action='store_true', default=False, help='use likely misclassified points as test points')
    parser.add_argument('--misclassified_minority', action='store_true', default=False, help='use likely misclassified and minority class points as test points')
    parser.add_argument('--full_dataset', action='store_true', default=False, help='use full dataset to train model')
    parser.add_argument('--impl', default='ibm', help='which implementation to use')
    parser.add_argument('--sampling_type', default='over', help='over, under or smote sampling')
    parser.add_argument('--sampling_ratio', type=float, default=0.5, help='sampling ratio')
    args = parser.parse_args()

    print(vars(args))
   
    # Create experiment directory.
    experiment_path = f'/homes/al5217/PyStatDP/examples/{args.algorithm}/'
    experiment_number = len(os.listdir(experiment_path))
    print("experiment_number: {}".format(experiment_number))
    try:
        os.makedirs('{}experiment-{}'.format(experiment_path, experiment_number))
        print("Created directory.")
    except FileExistsError:
        print('error creating file :( current path: {}'.format(Path.cwd()))
        pass

    # Run experiment.
    psd = pystatdp()
    start = time.time()

    if args.algorithm == 'iris':
        dataset = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)
        
        if args.full_dataset:
            X_train = dataset.data
            y_train = dataset.target

        if args.misclassified:
            # Set X_test as likely misclassified points
            likely_misclassified = [70, 72, 83, 84, 106, 119, 133, 138]
            X_test = np.take(dataset.data, likely_misclassified, axis=0)

        lower = np.array([4.3, 2. , 1. , 0.1]) # lower bound of each feature's values
        upper = np.array([7.5, 4. , 6. , 2.]) # upper bound of each feature's values
        priors = np.array([0.5, 0.5, 0.5]) # priors of each classes
        probability = 0.002 # probability for geometric distribution
        var_smoothing = 1e-4 # variance smoothing

        algorithm = Iris
        algo_params = tuple((lower, upper, priors, probability, var_smoothing, X_train, y_train, X_test, args.pred_len, args.train_privacy, args.impl))

    elif args.algorithm == 'diabetes':
        dataset = datasets.load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(dataset.data[:, :2], dataset.target, test_size=0.2, random_state=42)
        
        # Join train and test sets.
        X_full = np.append(X_train, X_test, axis=0)
        y_full = np.append(y_train, y_test, axis=0)     
        
        bounds_X = (-0.138, 0.2)
        bounds_y = (25, 346)

        algorithm = Diabetes
        algo_params = tuple((bounds_X, bounds_y, X_train, y_train, X_test, args.pred_len, args.train_privacy))

    elif args.algorithm in ['adult', 'adult-resample']:
        X_train = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", usecols=(0, 4, 10, 11, 12), delimiter=", ")
        y_train = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", usecols=14, dtype=str, delimiter=", ")
        X_test = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", usecols=(0, 4, 10, 11, 12), delimiter=", ", skiprows=1)
        y_test = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", usecols=14, dtype=str, delimiter=", ", skiprows=1)
        # Must trim trailing period "." from label
        y_test = np.array([a[:-1] for a in y_test])

        # Convert labels to numerical values.
        y_train = np.where(y_train == '>50K', 1, 0)
        y_test = np.where(y_test == '>50K', 1, 0)
        
        # Join train and test sets.
        X_full = np.append(X_train, X_test, axis=0)
        y_full = np.append(y_train, y_test, axis=0)     
       
        lower = [17.,  1.,  0.,  0.,  1.]
        upper = [9.0000e+01, 1.6000e+01, 9.9999e+04, 4.3560e+03, 9.9000e+01]

        # Get minority samples from test set. ('>50K' is the minority class)
        minority_test_indices = np.where(y_full == 1)
        X_test = X_full[minority_test_indices]

        if args.misclassified_minority:
            likely_misclassified_minority = [10, 25, 27, 40995, 38, 24614, 24616, 32809, 8234, 41009, 8243, 16438, 8249, 8252, 63, 8262]
            X_test = np.take(X_full, likely_misclassified_minority, axis=0)
        elif args.misclassified:
            # Set X_test as likely misclassified points
            likely_misclassified = [6153, 6844, 455, 7335, 25, 7749, 16088, 27, 39862, 28, 5974, 4947, 8750, 9023, 11449, 38]
            X_test = np.take(X_full, likely_misclassified, axis=0)
            
        if args.algorithm == 'adult-resample':
            algorithm = AdultResample
            algo_params = tuple((args.sampling_type, args.sampling_ratio, lower, upper, X_full, y_full, X_test, args.pred_len, args.train_privacy, args.impl))
        else:
            algorithm = Adult
            algo_params = tuple((lower, upper, X_full, y_full, X_test, args.pred_len, args.train_privacy, args.impl))

    elif args.algorithm == 'pima':
        X_full = np.loadtxt("data/pima-indians-diabetes.csv", usecols=(0,1,2,3,4,5,6,7), delimiter=",")
        y_full = np.loadtxt("data/pima-indians-diabetes.csv", usecols=(8), delimiter=",")

        lower = [0,0,0,0,0,0,0.078,21]
        upper = [17,199,122,99,856,68,2.42,81]

        likely_misclassified = [4, 405, 6, 508, 7, 468, 8, 286, 9, 684, 12, 642, 13, 228]
        likely_misclassified_minority = [6, 9, 10, 522, 12, 524, 14, 16, 17, 18, 19, 20, 23, 25, 26, 539, 540, 31, 544]
        if args.misclassified_minority:
            X_test = np.take(X_full, likely_misclassified_minority, axis=0)
        elif args.misclassified:
            X_test = np.take(X_full, likely_misclassified, axis=0)
        else:
            X_test = X_full

        algorithm = Pima
        algo_params = tuple((lower, upper, X_full, y_full, X_test, args.pred_len, args.train_privacy, args.impl))

    elif args.algorithm in ['machine', 'machine-resample']:
        data = np.loadtxt('data/machine.data', usecols=(2,3,4,5,6,7,8), delimiter=',')
        X_full = np.loadtxt('data/machine.data', usecols=(2,3,4,5,6,7), delimiter=',')
        y_full = np.loadtxt('data/machine.data', usecols=(8), delimiter=',')

        lower = [17., 64., 64.,  0.,  0.,  0.]
        upper = [1.50e+03, 3.20e+04, 6.40e+04, 2.56e+02, 5.20e+01, 1.76e+02]

        outliers = [199, 2, 5, 6, 7, 8, 9, 30, 31, 35, 65, 94, 95, 96, 97, 150, 151, 152, 153, 154, 155, 156, 169, 191, 192, 195, 196, 197, 198, 1]
        X_test = np.take(X_full, outliers, axis=0)

        if args.algorithm == 'machine-resample':
            algorithm = MachineResample
            algo_params = tuple((args.sampling_type, args.sampling_ratio, lower, upper, X_full, y_full, X_test, args.pred_len, args.train_privacy, args.impl))
        else:
            algorithm = Machine
            algo_params = tuple((lower, upper, X_full, y_full, X_test, args.pred_len, args.train_privacy, args.impl))

    else:
        algorithm = Dummy
        algo_params = tuple((args.pred_len,))

    results = psd.main(algorithm, algo_params, tuple((args.privacy,)), e_iter=args.e_iter, d_iter=args.d_iter, test_range=args.test_range, n_checks=args.n_checks)

    # Plot and save to file
    plot_file = "{}experiment-{}/test_result.pdf".format(experiment_path, experiment_number)
    psd.plot_result(results, r'Test $\epsilon$', 'P Value', "{}-{} (e={})".format(args.algorithm, args.impl, args.train_privacy), plot_file)

    # Get measured epsilon and format results.
    # results[test_budget] = [(epsilon, p, d1, d2, kwargs, event)]
    results = results[args.privacy]
    measured_epsilon = -1.0
    p_values = []
    for (epsilon, p, d1, d2, kwargs, event) in reversed(results):
        p_values.append({'epsilon': epsilon, 'p_value': p})
        if p > args.threshold:
            measured_epsilon = epsilon

    results_json = {
        'experiment_args': vars(args),
        'experiment_time': time.time() - start,
        'measured_epsilon': measured_epsilon,
        'p_values': p_values,
        'results': encode(results, unpicklable=False)
    }

    # Dump the results to file
    json_file = Path.cwd() / f'{args.algorithm}/experiment-{experiment_number}/test_results.json'
    with json_file.open('w') as f:
        json.dump(results_json, f, indent="  ")
