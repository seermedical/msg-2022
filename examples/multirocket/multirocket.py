# Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb
#
# MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification
#
# Tan, Chang Wei, et al. "MultiRocket: Multiple pooling operators and transformations for fast and effective
# time series classification." Data Mining and Knowledge Discovery (2022): 1-24.
# https://doi.org/10.1007/s10618-022-00844-1
import os
import pickle
import time

import numba
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from numba import njit, prange
from p_tqdm import p_map
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from logistic_regression import LogisticRegression
from tools import load_data

name = "MultiRocket"


@njit("float32[:](float64[:,:,:],int32[:],int32[:],int32[:],int32[:],float32[:])",
      fastmath=True, parallel=False, cache=True)
def _fit_biases(X, num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, quantiles):
    num_examples, num_channels, input_length = X.shape

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0, 1, 6, 0, 1, 7, 0, 1, 8,
        0, 2, 3, 0, 2, 4, 0, 2, 5, 0, 2, 6, 0, 2, 7, 0, 2, 8, 0, 3, 4,
        0, 3, 5, 0, 3, 6, 0, 3, 7, 0, 3, 8, 0, 4, 5, 0, 4, 6, 0, 4, 7,
        0, 4, 8, 0, 5, 6, 0, 5, 7, 0, 5, 8, 0, 6, 7, 0, 6, 8, 0, 7, 8,
        1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2, 7, 1, 2, 8, 1, 3, 4,
        1, 3, 5, 1, 3, 6, 1, 3, 7, 1, 3, 8, 1, 4, 5, 1, 4, 6, 1, 4, 7,
        1, 4, 8, 1, 5, 6, 1, 5, 7, 1, 5, 8, 1, 6, 7, 1, 6, 8, 1, 7, 8,
        2, 3, 4, 2, 3, 5, 2, 3, 6, 2, 3, 7, 2, 3, 8, 2, 4, 5, 2, 4, 6,
        2, 4, 7, 2, 4, 8, 2, 5, 6, 2, 5, 7, 2, 5, 8, 2, 6, 7, 2, 6, 8,
        2, 7, 8, 3, 4, 5, 3, 4, 6, 3, 4, 7, 3, 4, 8, 3, 5, 6, 3, 5, 7,
        3, 5, 8, 3, 6, 7, 3, 6, 8, 3, 7, 8, 4, 5, 6, 4, 5, 7, 4, 5, 8,
        4, 6, 7, 4, 6, 8, 4, 7, 8, 5, 6, 7, 5, 6, 8, 5, 7, 8, 6, 7, 8
    ), dtype=np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype=np.float32)

    feature_index_start = 0

    combination_index = 0
    num_channels_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):

            feature_index_end = feature_index_start + num_features_this_dilation

            num_channels_this_combination = num_channels_per_combination[combination_index]

            num_channels_end = num_channels_start + num_channels_this_combination

            channels_this_combination = channel_indices[num_channels_start:num_channels_end]

            _X = X[np.random.randint(num_examples)][channels_this_combination]

            A = -_X  # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X

            C_alpha = np.zeros((num_channels_this_combination, input_length), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, num_channels_this_combination, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
            C = np.sum(C, axis=0)

            biases[feature_index_start:feature_index_end] = np.quantile(C, quantiles[
                                                                           feature_index_start:feature_index_end])

            feature_index_start = feature_index_end

            combination_index += 1
            num_channels_start = num_channels_end

    return biases


def _fit_dilations(input_length, num_features, max_dilations_per_kernel):
    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = \
        np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(np.int32),
                  return_counts=True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32)  # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32)


def fit(X, num_features=10_000, max_dilations_per_kernel=32):
    _, num_channels, input_length = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel)

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    num_dilations = len(dilations)
    num_combinations = num_kernels * num_dilations

    max_num_channels = min(num_channels, 9)
    max_exponent = np.log2(max_num_channels + 1)

    num_channels_per_combination = (2 ** np.random.uniform(0, max_exponent, num_combinations)).astype(np.int32)

    channel_indices = np.zeros(num_channels_per_combination.sum(), dtype=np.int32)

    num_channels_start = 0
    for combination_index in range(num_combinations):
        num_channels_this_combination = num_channels_per_combination[combination_index]
        num_channels_end = num_channels_start + num_channels_this_combination
        channel_indices[num_channels_start:num_channels_end] = np.random.choice(num_channels,
                                                                                num_channels_this_combination,
                                                                                replace=False)

        num_channels_start = num_channels_end

    biases = _fit_biases(X, num_channels_per_combination, channel_indices,
                         dilations, num_features_per_dilation, quantiles)

    return num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases


@njit(
    "float32[:,:](float64[:,:,:],float64[:,:,:],Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),int32)",
    fastmath=True, parallel=True, cache=True)
def transform(X, X1, parameters, parameters1, n_features_per_kernel=4):
    num_examples, num_channels, input_length = X.shape

    num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases = parameters
    _, _, dilations1, num_features_per_dilation1, biases1 = parameters1

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0, 1, 6, 0, 1, 7, 0, 1, 8,
        0, 2, 3, 0, 2, 4, 0, 2, 5, 0, 2, 6, 0, 2, 7, 0, 2, 8, 0, 3, 4,
        0, 3, 5, 0, 3, 6, 0, 3, 7, 0, 3, 8, 0, 4, 5, 0, 4, 6, 0, 4, 7,
        0, 4, 8, 0, 5, 6, 0, 5, 7, 0, 5, 8, 0, 6, 7, 0, 6, 8, 0, 7, 8,
        1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2, 7, 1, 2, 8, 1, 3, 4,
        1, 3, 5, 1, 3, 6, 1, 3, 7, 1, 3, 8, 1, 4, 5, 1, 4, 6, 1, 4, 7,
        1, 4, 8, 1, 5, 6, 1, 5, 7, 1, 5, 8, 1, 6, 7, 1, 6, 8, 1, 7, 8,
        2, 3, 4, 2, 3, 5, 2, 3, 6, 2, 3, 7, 2, 3, 8, 2, 4, 5, 2, 4, 6,
        2, 4, 7, 2, 4, 8, 2, 5, 6, 2, 5, 7, 2, 5, 8, 2, 6, 7, 2, 6, 8,
        2, 7, 8, 3, 4, 5, 3, 4, 6, 3, 4, 7, 3, 4, 8, 3, 5, 6, 3, 5, 7,
        3, 5, 8, 3, 6, 7, 3, 6, 8, 3, 7, 8, 4, 5, 6, 4, 5, 7, 4, 5, 8,
        4, 6, 7, 4, 6, 8, 4, 7, 8, 5, 6, 7, 5, 6, 8, 5, 7, 8, 6, 7, 8
    ), dtype=np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)
    num_dilations1 = len(dilations1)

    num_features = num_kernels * np.sum(num_features_per_dilation)
    num_features1 = num_kernels * np.sum(num_features_per_dilation1)

    features = np.zeros((num_examples, (num_features + num_features1) * n_features_per_kernel), dtype=np.float32)
    n_features_per_transform = np.int64(features.shape[1] / 2)

    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        # Base series
        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros((num_channels, input_length), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, num_channels, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                num_channels_this_combination = num_channels_per_combination[combination_index]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[num_channels_start:num_channels_end]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha[channels_this_combination] + \
                    C_gamma[index_0][channels_this_combination] + \
                    C_gamma[index_1][channels_this_combination] + \
                    C_gamma[index_2][channels_this_combination]
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                else:
                    _c = C[padding:-padding]

                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += _c[j] + _bias
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / _c.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1

                feature_index_start = feature_index_end

                combination_index += 1
                num_channels_start = num_channels_end

        # First order difference
        _X1 = X1[example_index]
        A1 = -_X1  # A = alpha * X = -X
        G1 = _X1 + _X1 + _X1  # G = gamma * X = 3X

        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations1):

            _padding0 = dilation_index % 2

            dilation = dilations1[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation1[dilation_index]

            C_alpha = np.zeros((num_channels, input_length - 1), dtype=np.float32)
            C_alpha[:] = A1

            C_gamma = np.zeros((9, num_channels, input_length - 1), dtype=np.float32)
            C_gamma[9 // 2] = G1

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A1[:, :end]
                C_gamma[gamma_index, :, -end:] = G1[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A1[:, start:]
                C_gamma[gamma_index, :, :-start] = G1[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                num_channels_this_combination = num_channels_per_combination[combination_index]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[num_channels_start:num_channels_end]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha[channels_this_combination] + \
                    C_gamma[index_0][channels_this_combination] + \
                    C_gamma[index_1][channels_this_combination] + \
                    C_gamma[index_2][channels_this_combination]
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index + n_features_per_transform
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                else:
                    _c = C[padding:-padding]

                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += _c[j] + _bias
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index + n_features_per_transform
                        features[example_index, end] = ppv / _c.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1

                feature_index_start = feature_index_end

    return features


class MultiRocketTransform:
    def __init__(
            self,
            num_features=50000,
            max_dilations_per_kernel=32,
            random_state=None,
            num_threads=-1
    ):
        """MultiRocket
        Multiple pooling operators and transformations for fast and effective time series classification

        **Multivariate**
        MultiVariate input only.  Use class MultiRocket for univariate input.
        @article{tan_et_al_2022,
          title     = {{MultiRocket}: Multiple pooling operators and transformations for fast and effective time series classification},
          author    = {Tan, Chang Wei and Dempster, Angus and Bergmeir, Christoph and Webb, Geoffrey I},
          journal   = {Data Mining and Knowledge Discovery},
          pages     = {1--24},
          year      = {2022},
          publisher = {Springer}
        }
        Parameters
        ----------
        num_features             : int, number of features (default 50,000)
        max_dilations_per_kernel : int, maximum number of dilations per kernel (default 32)
        random_state             : int, random seed (optional, default None)
        num_threads              : int, number of threads (optional, default -1)
        """
        if num_threads < 0:
            num_threads = psutil.cpu_count(logical=True)
            numba.set_num_threads(num_threads)
        else:
            numba.set_num_threads(min(num_threads, psutil.cpu_count(logical=True)))

        self.base_parameters = None
        self.diff1_parameters = None

        self.n_features_per_kernel = 4
        self.num_features = num_features / 2  # 1 per transformation
        self.num_kernels = int(self.num_features / self.n_features_per_kernel)

        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = (
            np.int32(random_state) if isinstance(random_state, int) else None
        )

        self._is_fitted = False

    def fit(self, X, y=None):
        """Fits dilations and biases to input time series.
        Parameters
        ----------
        X : pandas DataFrame, input time series (sktime format)
        y : array_like, target values (optional, ignored as irrelevant)
        Returns
        -------
        self
        """
        if X.shape[2] < 10:
            # handling very short series (like PensDigit from the MTSC archive)
            # series have to be at least a length of 10 (including differencing)
            _x_train = np.zeros((X.shape[0], X.shape[1], 10), dtype=X.dtype)
            _x_train[:, :, :X.shape[2]] = X
            X = _x_train
            del _x_train

        xx = np.diff(X, 1)

        self.base_parameters = fit(
            X,
            num_features=self.num_kernels
        )
        self.diff1_parameters = fit(
            xx,
            num_features=self.num_kernels
        )

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Transforms input time series.
        Parameters
        ----------
        X : pandas DataFrame, input time series (sktime format)
        y : array_like, target values (optional, ignored as irrelevant)
        Returns
        -------
        pandas DataFrame, transformed features
        """
        if X.shape[2] < 10:
            # handling very short series (like PensDigit from the MTSC archive)
            # series have to be at least a length of 10 (including differencing)
            _x_train = np.zeros((X.shape[0], X.shape[1], 10), dtype=X.dtype)
            _x_train[:, :, :X.shape[2]] = X
            X = _x_train
            del _x_train

        xx = np.diff(X, 1)

        return pd.DataFrame(transform(
            X, xx,
            self.base_parameters, self.diff1_parameters,
            self.n_features_per_kernel
        ))

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.
        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.
        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to be transformed
        y : pd.Series or np.ndarray, optional (default=None)
            Target values of data to be transformed.
        Returns
        -------
        Xt : pd.DataFrame, pd.Series or np.ndarray
            Transformed data.
        """
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        if X is None:
            # Fit method of arity 1 (unsupervised transformation)
            return self.fit(X).transform(X)
        else:
            # Fit method of arity 2 (supervised transformation)
            return self.fit(X, y).transform(X)


def fit_multirocket(
        x, num_features: int = 50_000, max_dilations: int = 32
) -> (np.ndarray, MultiRocketTransform):
    rocket = MultiRocketTransform(
        num_features, max_dilations_per_kernel=max_dilations, random_state=42
    )
    X_train_transform = rocket.fit_transform(x)

    return X_train_transform, rocket


class MultiRocket:
    def __init__(
            self,
            num_features=50000,
            max_dilations=32,
            max_epochs=200,
            classifier="ridge",
            num_threads=-1,
            verbose=0,
            save_path=None,
            load_model=False,
    ):
        if num_threads < 0:
            num_threads = psutil.cpu_count(logical=True)
        else:
            num_threads = min(num_threads, psutil.cpu_count(logical=True))

        numba.set_num_threads(num_threads)
        self.num_threads = num_threads

        self.save_path = save_path
        self.load_model = load_model
        self.classifier_type = classifier

        self.name = name

        self.base_parameters = None
        self.diff1_parameters = None

        self.max_epochs = max_epochs
        self.max_dilations = max_dilations
        self.n_features_per_kernel = 4
        self.num_features = num_features / 2  # 1 per transformation
        self.num_kernels = int(self.num_features / self.n_features_per_kernel)

        if verbose > 1:
            print('[{}] Creating {} with {} kernels'.format(self.name, self.name, self.num_kernels))

        if classifier.lower() == "ridge":
            self.classifier = make_pipeline(
                StandardScaler(),
                RidgeClassifierCV(
                    alphas=np.logspace(-3, 3, 10),
                    normalize=False
                )
            )
        else:
            self.classifier = LogisticRegression(
                num_features=num_features,
                max_epochs=max_epochs,
                save_path=save_path
            )

        self.stats = {
            "model_name": name,
            "train_acc": -1,
            "train_duration": 0,
            "test_duration": 0,
            "generate_kernel_duration": 0,
            "train_transforms_duration": 0,
            "test_transforms_duration": 0,
            "apply_kernel_on_train_duration": 0,
            "apply_kernel_on_test_duration": 0,
        }

        self.verbose = verbose

        self.multirocket = None

    def fit(
            self,
            x_train, y_train,
            predict_on_train=True,
            **kwargs
    ):
        if self.verbose > 1:
            print('[{}] Training with training set of {}'.format(self.name, x_train.shape))

        if x_train.shape[2] < 10:
            # handling very short series (like PensDigit from the MTSC archive)
            # series have to be at least a length of 10 (including differencing)
            _x_train = np.zeros((x_train.shape[0], x_train.shape[1], 10), dtype=x_train.dtype)
            _x_train[:, :, :x_train.shape[2]] = x_train
            x_train = _x_train
            del _x_train

        start_time = time.perf_counter()

        _start_time = time.perf_counter()
        xx = np.diff(x_train, 1)
        train_transforms_duration = time.perf_counter() - _start_time

        _start_time = time.perf_counter()
        self.base_parameters = fit(
            x_train,
            num_features=self.num_kernels
        )
        self.diff1_parameters = fit(
            xx,
            num_features=self.num_kernels
        )
        generate_kernel_duration = time.perf_counter() - _start_time

        _start_time = time.perf_counter()
        x_train_transform = transform(
            x_train, xx,
            self.base_parameters, self.diff1_parameters,
            self.n_features_per_kernel
        )
        apply_kernel_on_train_duration = time.perf_counter() - _start_time

        x_train_transform = np.nan_to_num(x_train_transform)

        elapsed_time = time.perf_counter() - start_time
        if self.verbose > 1:
            print('[{}] Kernels applied!, took {}s'.format(self.name, elapsed_time))
            print('[{}] Transformed Shape {}'.format(self.name, x_train_transform.shape))

        if self.verbose > 1:
            print('[{}] Training'.format(self.name))

        _start_time = time.perf_counter()
        self.classifier.fit(x_train_transform, y_train)
        train_duration = time.perf_counter() - _start_time

        if self.verbose > 1:
            print('[{}] Training done!, took {:.3f}s'.format(self.name, train_duration))
        if predict_on_train:
            yhat = self.classifier.predict(x_train_transform)
        else:
            yhat = None

        self.stats.update({
            "generate_kernel_duration": generate_kernel_duration,
            "train_transforms_duration": train_transforms_duration,
            "apply_kernel_on_train_duration": apply_kernel_on_train_duration,
            "train_duration": train_duration,
        })

        return yhat

    def fit_large(
            self,
            sampled_data,
            predict_on_train=True,
            **kwargs
    ):
        X_train = []
        X_validation = []
        y_train = []
        y_validation = []
        for patient, group in sampled_data.groupby("patient"):
            _x_train, _x_validation, _y_train, _y_validation = train_test_split(
                group["filepath"],
                group["label"],
                test_size=0.2,
                random_state=42,
            )
            X_train.append(_x_train)
            X_validation.append(_x_validation)
            y_train.append(_y_train)
            y_validation.append(_y_validation)

        X_train = pd.concat(X_train)
        X_validation = pd.concat(X_validation)
        y_train = pd.concat(y_train)
        y_validation = pd.concat(y_validation)

        if self.verbose > 1:
            print('[{}] Training with training set of {}'.format(self.name, X_train.shape[0]))

        pos_samples = X_train[y_train == 1]
        pos_samples = pos_samples.sample(int(0.2 * pos_samples.shape[0]))

        neg_samples = X_train[y_train == 0]
        neg_samples = neg_samples.sample(int(0.1 * neg_samples.shape[0]))

        tmp = pd.concat([pos_samples, neg_samples]).reset_index(drop=True)
        if self.verbose > 1:
            print(f"Fitting multirocket with {tmp.shape[0]} series")

        sampled_data = np.array(p_map(
            load_data,
            tmp,
            num_cpus=self.num_threads
        ))
        del tmp

        transformed_data, multirocket = fit_multirocket(
            sampled_data,
            num_features=int(self.num_features * 2),
            max_dilations=self.max_dilations
        )
        del sampled_data

        self.multirocket = multirocket

        def multirocket_transform(x):
            x = x.numpy()
            x_shape = x.shape
            if len(x_shape) == 2:
                x = np.reshape(x, (1, x.shape[0], x.shape[1]))
            transformed_x = multirocket.transform(x)

            if len(x_shape) == 2:
                transformed_x = transformed_x.values.flatten()

            return transformed_x

        def tf_multirocket_transform(x, y=None):
            x = tf.py_function(multirocket_transform, [x], tf.float64)
            if y is None:
                return x
            return x, y

        self.classifier = LogisticRegression(
            num_features=self.num_features,
            max_epochs=self.max_epochs,
            save_path=self.save_path
        )
        _start_time = time.perf_counter()
        self.classifier.fit_large(
            X_train, y_train,
            X_validation, y_validation,
            transform_func=tf_multirocket_transform,
            dim=transformed_data.shape[-1]
        )
        train_duration = time.perf_counter() - _start_time

        if self.verbose > 1:
            print('[{}] Training done!, took {:.3f}s'.format(self.name, train_duration))
        if predict_on_train:
            yhat = self.classifier.predict_large(X_train, transform_func=tf_multirocket_transform)
        else:
            yhat = None

        self.stats.update({
            "train_duration": train_duration,
        })

        return yhat

    def predict(self, x):
        if self.verbose > 1:
            print('[{}] Predicting'.format(self.name))

        _start_time = time.perf_counter()
        xx = np.diff(x, 1)
        test_transforms_duration = time.perf_counter() - _start_time

        _start_time = time.perf_counter()
        x_transform = transform(
            x, xx,
            self.base_parameters, self.diff1_parameters,
            self.n_features_per_kernel
        )
        apply_kernel_on_test_duration = time.perf_counter() - _start_time

        x_transform = np.nan_to_num(x_transform)
        if self.verbose > 1:
            print('Kernels applied!, took {:.3f}s. Transformed shape: {}.'.format(
                apply_kernel_on_test_duration,
                x_transform.shape))

        start_time = time.perf_counter()
        yhat = self.classifier.predict(x_transform)
        test_duration = time.perf_counter() - start_time
        if self.verbose > 1:
            print("[{}] Predicting completed, took {:.3f}s".format(self.name, test_duration))

        self.stats.update({
            "test_transforms_duration": test_transforms_duration,
            "apply_kernel_on_test_duration": apply_kernel_on_test_duration,
            "test_duration": test_duration,
        })

        if x.shape[0] == 1:
            return yhat[0][0]
        return yhat

    def predict_large(self, x):
        if self.verbose > 1:
            print('[{}] Predicting'.format(self.name))

        def multirocket_transform(xx):
            xx = xx.numpy()
            x_shape = xx.shape
            if len(x_shape) == 2:
                xx = np.reshape(xx, (1, xx.shape[0], xx.shape[1]))
            transformed_x = self.multirocket.transform(xx)

            if len(x_shape) == 2:
                transformed_x = transformed_x.values.flatten()

            return transformed_x

        def tf_multirocket_transform(xx):
            xx = tf.py_function(multirocket_transform, [xx], tf.float64)
            return xx

        start_time = time.perf_counter()
        yhat = self.classifier.predict_large(x, transform_func=tf_multirocket_transform)
        test_duration = time.perf_counter() - start_time

        if self.verbose > 1:
            print("[{}] Predicting completed, took {:.3f}s".format(self.name, test_duration))

        self.stats.update({
            "test_duration": test_duration,
        })

        if x.shape[0] == 1:
            return yhat[0][0]
        return yhat

    def save(self):
        if self.classifier_type.lower() == "logistic":
            self.classifier.model.save(self.save_path)
            self.classifier.model = None

        file = open(self.save_path + "/model." + self.name + '.pkl', 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

        if self.multirocket is not None:
            file = open(self.save_path + "/" + self.name + '.pkl', 'wb')
            file.write(pickle.dumps(self.multirocket))
            file.close()

    def load(self):
        save_path = self.save_path
        filename = save_path + "/model." + self.name + '.pkl'
        if os.path.exists(filename):
            file = open(filename, 'rb')
            data_pickle = file.read()
            file.close()

            self.__dict__ = pickle.loads(data_pickle)

            if self.classifier_type.lower() == "logistic":
                self.classifier.model = tf.keras.models.load_model(save_path)
            return

        filename = save_path + "/" + self.name + '.pkl'
        if os.path.exists(filename):
            file = open(filename, 'rb')
            data_pickle = file.read()
            file.close()

            self.multirocket = pickle.loads(data_pickle)
