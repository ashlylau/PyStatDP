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
from itertools import zip_longest

import numpy as np
import pydp as dp


algo_dict = {
    'BoundedMean': 'example call with parameters: dp.BoundedMean(epsilon, -15, 15)',
    'Max': 'example call with parameters: dp.Max(epsilon)',
    'BoundedStandardDeviation': 'example call with parameters: dp.BoundedStandardDeviation(epsilon, 0, 15)',
    'BoundedSum': 'example call with parameters: dp.BoundedSum(epsilon, 0, 10)',
    'BoundedVariance': 'example call with parameters: dp.BoundedVariance(epsilon, 0, 16)',
    'Median': 'example call with parameters: dp.Median(epsilon)',
    'Percentile': 'example call with parameters: dp.Percentile(epsilon)'
}


def generic_method(prng, queries, epsilon, algorithm, param_for_algorithm):
    '''
    A generic method to route incoming tasks.
    param prng: Psuedo random number generator, ! to be removed.
    param queries: queries to the algorithm
    param epsilon: privacy budget
    param algorithm: The algorithm to be tested; (e.g dp.BoundedMean, dp.BoundedSum)
    param param_to_algorithm (a tuple): inputs to the algortihm. 

    prng = 2
    queries = [1,2,3,4,5]
    print(generic_method(prng, queries, 1.0, dp.BoundedMean, (-15,15)))
    >>> example call with parameters: dp.BoundedMean(epsilon, -15, 15)
        0.0
    '''

    print(algo_dict[str(algorithm)[13:-2]])
    return algorithm(epsilon, *param_for_algorithm).result(queries) # , epsilon)


# def dp_mean(prng, queries, epsilon):
#     # PyDP mean
#     x = dp.BoundedMean(epsilon, -15, 15)
#     return x.result(queries)


# def dp_max(prng, queries, epsilon):
#     x = dp.Max(epsilon)
#     return x.result(queries, epsilon)


# # dict to get all the parameters
# def dp_bounded_standard_deviation(prng, queries, epsilon):
#     # INCORRECT, issue with params
#     return dp.BoundedStandardDeviation(epsilon, 0, 15).result(queries)


# def dp_bounded_sum(prng, queries, epsilon):
#     return dp.BoundedSum(epsilon, 0, 10).result(queries)


# def dp_bounded_variance(prng, queries, epsilon):
#     return dp.BoundedVariance(epsilon, 0, 16).result(queries)


# def dp_median(prng, queries, epsilon):
#     x= dp.Median(epsilon)
#     return x.result(queries, epsilon)


# def dp_percentile(prng, queries, epsilon):
#     return dp.Percentile(epsilon).result(queries, epsilon)


# def _hamming_distance(result1, result2):
#     # implement hamming distance in pure python, faster than np.count_zeros if inputs are plain python list
#     return sum(res1 != res2 for res1, res2 in zip_longest(result1, result2))


# def noisy_max_v1a(prng, queries, epsilon):
#     # find the largest noisy element and return its index
#     prng = np.random.default_rng()
#     return (np.asarray(queries, dtype=np.float64) + prng.laplace(scale=2.0 / epsilon, size=len(queries))).argmax()


# def noisy_max_v1b(prng, queries, epsilon):
#     # INCORRECT: returning maximum value instead of the index
#     return (np.asarray(queries, dtype=np.float64) + prng.laplace(scale=2.0 / epsilon, size=len(queries))).max()


# def noisy_max_v2a(prng, queries, epsilon):
#     return (np.asarray(queries, dtype=np.float64) + prng.exponential(scale=2.0 / epsilon, size=len(queries))).argmax()


# def noisy_max_v2b(prng, queries, epsilon):
#     # INCORRECT: returning the maximum value instead of the index
#     return (np.asarray(queries, dtype=np.float64) + prng.exponential(scale=2.0 / epsilon, size=len(queries))).max()


# def histogram_eps(prng, queries, epsilon):
#     # INCORRECT: using (epsilon) noise instead of (1 / epsilon)
#     noisy_array = np.asarray(queries, dtype=np.float64) + \
#         prng.laplace(scale=epsilon, size=len(queries))
#     return noisy_array[0]


# def histogram(prng, queries, epsilon):
#     noisy_array = np.asarray(queries, dtype=np.float64) + \
#         prng.laplace(scale=1.0 / epsilon, size=len(queries))
#     return noisy_array[0]


# def SVT(prng, queries, epsilon, N, T):
#     out = []
#     eta1 = prng.laplace(scale=2.0 / epsilon)
#     noisy_T = T + eta1
#     c1 = 0
#     for query in queries:
#         eta2 = prng.laplace(scale=4.0 * N / epsilon)
#         if query + eta2 >= noisy_T:
#             out.append(True)
#             c1 += 1
#             if c1 >= N:
#                 break
#         else:
#             out.append(False)
#     return out.count(False)


# def iSVT1(prng, queries, epsilon, N, T):
#     out = []
#     eta1 = prng.laplace(scale=2.0 / epsilon)
#     noisy_T = T + eta1
#     for query in queries:
#         # INCORRECT: no noise added to the queries
#         eta2 = 0
#         if (query + eta2) >= noisy_T:
#             out.append(True)
#         else:
#             out.append(False)

#     true_count = int(len(queries) / 2)
#     return _hamming_distance((True if i < true_count else False for i in range(len(queries))), out)


# def iSVT2(prng, queries, epsilon, N, T):
#     out = []
#     eta1 = prng.laplace(scale=2.0 / epsilon)
#     noisy_T = T + eta1
#     for query in queries:
#         # INCORRECT: noise added to queries doesn't scale with N
#         eta2 = prng.laplace(scale=2.0 / epsilon)
#         if (query + eta2) >= noisy_T:
#             out.append(True)
#             # INCORRECT: no bounds on the True's to output
#         else:
#             out.append(False)

#     true_count = int(len(queries) / 2)
#     return _hamming_distance((True if i < true_count else False for i in range(len(queries))), out)


# def iSVT3(prng, queries, epsilon, N, T):
#     out = []
#     eta1 = prng.laplace(scale=4.0 / epsilon)
#     noisy_T = T + eta1
#     c1 = 0
#     for query in queries:
#         # INCORRECT: noise added to queries doesn't scale with N
#         eta2 = prng.laplace(scale=4.0 / (3.0 * epsilon))
#         if query + eta2 > noisy_T:
#             out.append(True)
#             c1 += 1
#             if c1 >= N:
#                 break
#         else:
#             out.append(False)

#     true_count = int(len(queries) / 2)
#     return _hamming_distance((True if i < true_count else False for i in range(len(queries))), out)


# def iSVT4(prng, queries, epsilon, N, T):
#     out = []
#     eta1 = prng.laplace(scale=2.0 / epsilon)
#     noisy_T = T + eta1
#     c1 = 0
#     for query in queries:
#         eta2 = prng.laplace(scale=2.0 * N / epsilon)
#         if query + eta2 > noisy_T:
#             # INCORRECT: Output the noisy query instead of True
#             out.append(query + eta2)
#             c1 += 1
#             if c1 >= N:
#                 break
#         else:
#             out.append(False)
#     return out.count(False), out[-1]
