# MIT License
#
# Copyright (c) 2018 Yuxin Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
from textwrap import dedent
import time
import json
from jsonpickle import encode
from pathlib import Path
import coloredlogs
from logging import getLogger
import matplotlib
import matplotlib.pyplot as plt
import pydp as dp
from statdp import detect_counterexample, ONE_DIFFER, ALL_DIFFER
from statdp.algorithms import dp_mean, dp_bounded_standard_deviation, dp_bounded_sum, \
    dp_bounded_variance, dp_max, dp_min, dp_median, dp_percentile, generic_method
# noisy_max_v1a, noisy_max_v1b, noisy_max_v2a, noisy_max_v2b, SVT, iSVT1, iSVT2, iSVT3, iSVT4, histogram, histogram_eps,
# switch matplotlib backend for running in background
matplotlib.use('agg')
matplotlib.rcParams['xtick.labelsize'] = '12'
matplotlib.rcParams['ytick.labelsize'] = '12'

coloredlogs.install(
    'INFO', fmt='%(asctime)s [0x%(process)x] %(levelname)s %(message)s')
logger = getLogger(__name__)


def plot_result(data, xlabel, ylabel, title, output_filename):
    """plot the results similar to the figures in our paper
    :param data: The input data sets to plots. e.g., {algorithm_epsilon: [(test_epsilon, pvalue), ...]}
    :param xlabel: The label for x axis.
    :param ylabel: The label for y axis.
    :param title: The title of the figure.
    :param output_filename: The output file name.
    :return: None
    """
    # setup the figure
    plt.ylim(0.0, 1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # colors and markers for each claimed epsilon
    markers = ['s', 'o', '^', 'x', '*', '+', 'p']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # add an auxiliary line for p-value=0.05
    plt.axhline(y=0.05, color='black', linestyle='dashed', linewidth=1.2)
    for i, (epsilon, points) in enumerate(data.items()):
        # add an auxiliary vertical line for the claimed privacy
        plt.axvline(x=float(epsilon), color=colors[i % len(
            colors)], linestyle='dashed', linewidth=1.2)
        # plot the
        x = [item[0] for item in points]
        p = [item[1] for item in points]
        plt.plot(x, p, 'o-',
                 label=f'$\\epsilon_0$ = {epsilon}', markersize=8, marker=markers[i % len(markers)], linewidth=3)

    # plot legends
    legend = plt.legend()
    legend.get_frame().set_linewidth(0.0)

    # save the figure and clear the canvas for next draw
    plt.savefig(output_filename, bbox_inches='tight')
    plt.gcf().clear()


def main(algo, param, epsilon):
    # list of tasks to test, each tuple contains (function, extra_args, sensitivity)
    tasks = [
        (generic_method, {'algorithm': algo,
                          'param_for_algorithm': param}, ALL_DIFFER)
        # (dp_mean, {}, ALL_DIFFER),
        # (dp_bounded_standard_deviation,  {}, ALL_DIFFER),
        # (dp_bounded_sum,  {}, ALL_DIFFER),
        # (dp_bounded_variance,  {}, ALL_DIFFER),
        # (dp_max, {}, ALL_DIFFER),
        # (dp_min, {}, ALL_DIFFER),
        # (dp_median,  {}, ALL_DIFFER),
        # (dp_percentile,  {}, ALL_DIFFER),
        # (noisy_max_v1a, {}, ALL_DIFFER),
        # (noisy_max_v1b, {}, ALL_DIFFER),
        # (noisy_max_v2a, {}, ALL_DIFFER),
        # (noisy_max_v2b, {}, ALL_DIFFER),
        # (histogram, {}, ONE_DIFFER),
        # (histogram_eps, {}, ONE_DIFFER),
        # (SVT, {'N': 1, 'T': 0.5}, ALL_DIFFER),
        # (iSVT1, {'T': 1, 'N': 1}, ALL_DIFFER),
        # (iSVT2, {'T': 1, 'N': 1}, ALL_DIFFER),
        # (iSVT3, {'T': 1, 'N': 1}, ALL_DIFFER),
        # (iSVT4, {'T': 1, 'N': 1}, ALL_DIFFER)
    ]

    # claimed privacy level to check
    # claimed_privacy = (0.9,)  # alter these values
    # claimed_privacy = np.linspace(.1,.9,5)
    claimed_privacy = epsilon

    # privacy levels to test, here we test from a range of 0.1 - 1.0 with a stepping of 0.1
    # test_privacy = tuple(x / 10.0 for x in range(1, 3, 1))
    test_privacy = tuple((0.9-0.09, 0.9, 0.9+0.09))

    for i, (algorithm, kwargs, sensitivity) in enumerate(tasks):
        start_time = time.time()
        results = {}
        for privacy_budget in claimed_privacy:
            # # privacy levels to test, here we test the claimed privacy plus .01 above and below
            # test_privacy = (privacy_budget -.09, privacy_budget, privacy_budget + .09)
            # set the third argument of the function (assumed to be `epsilon`) to the claimed privacy level
            kwargs[algorithm.__code__.co_varnames[1]] = privacy_budget
            results[privacy_budget] = detect_counterexample(
                algorithm, test_privacy, kwargs, sensitivity=sensitivity)

        # dump the results to file
        json_file = Path.cwd() / f'{algorithm.__name__}.json'
        if json_file.exists():
            logger.warning(
                f'{algorithm.__name__}.json already exists, note that it will be over-written')
            json_file.unlink()

        with json_file.open('w') as f:
            json.dump(encode(results, unpicklable=False), f)

        # plot and save to file
        plot_file = Path.cwd() / f'{algorithm.__name__}.pdf'
        if plot_file.exists():
            logger.warning(
                f'{algorithm.__name__}.pdf already exists, it will be over-written')
            plot_file.unlink()

        plot_result(results, r'Test $\epsilon$', 'P Value',
                    algorithm.__name__.replace('_', ' ').title(), plot_file)

        total_time, total_detections = time.time() - start_time, len(claimed_privacy) * \
            len(test_privacy)
        logger.info(f'[{i + 1} / {len(tasks)}]: {algorithm.__name__} | Time elapsed: {total_time:5.3f}s | '
                    f'Average time per detection: {total_time / total_detections:5.3f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="benchmark.py", formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=dedent('''\
        Validation of differential privacy standards!
        ---------------------------------------------
            Currently supported PyDP mechanisms:
            [Bounded Functions]
                1. BoundedMean (eg: BoundedMean(epsilon, -15, 15))
                2. BoundedStandardDeviation (eg: BoundedStandardDeviation(epsilon, 0, 15))
                3. BoundedSum (eg: BoundedSum(epsilon, 0, 10))
                4. BoundedVariance (eg: BoundedVariance(epsilon, 0, 16))
            [Order Statistics]
                1. Max (eg: Max(epsilon))
                2. Min (eg: Min(epsilon))
                3. Median (eg: Median(epsilon))
                4. Percentile (eg: Percentile(epsilon))

            ARGS:
            ------------------------------------------
                mechanism: algorithm to  validate
                param_for_mechanism: a tuple, parameters to the algorithm to validate.
                
            OUTPUT:
            ------------------------------------------
                p-value: a probabilistic estimate of how unlikely it is that
                    the null hypothesis(statistical hypothesis that we are trying to 
                    disprove.) is true.
         '''))
    parser.add_argument('--mechanism', metavar='mechanism',
                        help='a differential privacy mechanism')
    parser.add_argument('--param_for_mechanism', metavar='param_for_mechanism', type=int, nargs=2,
                        help='a tuple, parameters to the algorithm to validate.')
    parser.add_argument('--epsilon', type=float,
                        nargs='+', help='privacy budget')

    args = parser.parse_args()
    main(getattr(dp, args.mechanism), tuple(
        args.param_for_mechanism), tuple(args.epsilon))
