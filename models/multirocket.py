# Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb
#
# MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification
# https://arxiv.org/abs/2102.00457

import time

import numba
import numpy as np
import psutil
import tensorflow as tf
from numba import njit, prange
from sklearn.linear_model import RidgeClassifierCV
from sklearn.utils import shuffle

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


class MultiRocket:
    def __init__(
            self,
            num_features=50000,
            num_threads=-1,
            verbose=0
    ):
        if num_threads < 0:
            num_threads = psutil.cpu_count(logical=True)
            numba.set_num_threads(num_threads)
        else:
            numba.set_num_threads(min(num_threads, psutil.cpu_count(logical=True)))

        self.name = name

        self.base_parameters = None
        self.diff1_parameters = None

        self.n_features_per_kernel = 4
        self.num_features = num_features / 2  # 1 per transformation
        self.num_kernels = int(self.num_features / self.n_features_per_kernel)

        if verbose > 1:
            print('[{}] Creating {} with {} kernels'.format(self.name, self.name, self.num_kernels))

        self.classifier = RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 10),
            normalize=True
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

    def predict(self, x):
        if self.verbose > 1:
            print('[{}] Predicting'.format(self.name))

        if x.shape[2] < 10:
            # handling very short series (like PensDigit from the MTSC archive)
            # series have to be at least a length of 10 (including differencing)
            _x = np.zeros((x.shape[0], x.shape[1], 10), dtype=x.dtype)
            _x[:, :, :x.shape[2]] = x
            x = _x
            del _x

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

        return yhat, None


class MultiRocket_LR:
    def __init__(
            self,
            num_features=50000,
            num_threads=-1,
            verbose=0
    ):
        if num_threads < 0:
            num_threads = psutil.cpu_count(logical=True)
            numba.set_num_threads(num_threads)
        else:
            numba.set_num_threads(min(num_threads, psutil.cpu_count(logical=True)))

        self.name = name

        self.base_parameters = None
        self.diff1_parameters = None

        self.n_features_per_kernel = 4
        self.num_features = num_features / 2  # 1 per transformation
        self.num_kernels = int(self.num_features / self.n_features_per_kernel)

        if verbose > 1:
            print('[{}] Creating {} with {} kernels'.format(self.name, self.name, self.num_kernels))

        self.classifier = RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 10),
            normalize=True
        )

        self.args = None

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

        training_size = x_train.shape[0]

        args = {
            "num_features": self.num_features,
            "validation_size": 2 ** 11,
            "chunk_size": 2 ** 12,
            "minibatch_size": 256,
            "lr": 1e-4,
            "max_epochs": 50,
            "patience_lr": 5,  # 50 minibatches
            "patience": 10,  # 100 minibatches
            "cache_size": training_size  # set to 0 to prevent caching
        }

        a = 84 * 2 * 4
        _num_features = int(a * (args["num_features"] // a))
        del a
        num_chunks = np.int32(np.ceil(training_size / args["chunk_size"]))
        num_classes = len(np.unique(y_train))

        start_time = time.perf_counter()

        # -- cache -----------------------------------------------------------------

        # cache as much as possible to avoid unecessarily repeating the transform
        # consider caching to disk if appropriate, along the lines of numpy.memmap

        # -- model -----------------------------------------------------------------
        input_layer = tf.keras.layers.Input((_num_features,))
        output_layer = tf.keras.layers.Dense(
            num_classes,
            activation="softmax",
            bias_initializer="zeros",
            kernel_initializer="zeros",
        )(input_layer)
        model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer,
            name=self.name
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=args["lr"]),
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ]
        )
        # Instantiate an optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=args["lr"])
        # Instantiate a loss function
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5, min_lr=1e-8,
            patience=args["patience_lr"]
        )

        # -- validation data -------------------------------------------------------
        x_train, y_train = shuffle(x_train, y_train, random_state=0)
        x_training = x_train[args["validation_size"]:]
        y_training = y_train[args["validation_size"]:]
        x_validation = x_train[:args["validation_size"]]
        y_validation = y_train[:args["validation_size"]]

        # -- run -------------------------------------------------------------------\
        stop = False

        train_transforms_duration = 0
        val_transforms_duration = 0
        generate_kernel_duration = 0
        apply_kernel_on_train_duration = 0
        apply_kernel_on_validation_duration = 0

        _start_time = time.perf_counter()
        xx_train = np.diff(x_training, 1)
        train_transforms_duration += time.perf_counter() - _start_time

        _start_time = time.perf_counter()
        xx_val = np.diff(x_validation, 1)
        val_transforms_duration += time.perf_counter() - _start_time

        _start_time = time.perf_counter()
        base_parameters = fit(
            x_training,
            num_features=self.num_kernels
        )
        diff1_parameters = fit(
            xx_train,
            num_features=self.num_kernels
        )
        generate_kernel_duration += time.perf_counter() - _start_time

        # Prepare the training dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices((x_training, xx_train, y_training))
        train_dataset = train_dataset.shuffle(buffer_size=num_chunks).batch(args["minibatch_size"])

        # Prepare the validation dataset.
        val_dataset = tf.data.Dataset.from_tensor_slices((x_validation, xx_val, y_validation))
        val_dataset = val_dataset.batch(args["minibatch_size"])

        # Prepare the metrics.
        train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

        print("Training...")
        train_cache = {}
        val_cache = {}
        for epoch in range(args["max_epochs"]):
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, xx_batch_train, y_batch_train) in enumerate(train_dataset):

                if step not in train_cache.keys():
                    _start_time = time.perf_counter()
                    x_train_transform = transform(
                        x_batch_train, xx_batch_train,
                        base_parameters, diff1_parameters,
                        self.n_features_per_kernel
                    )
                    apply_kernel_on_train_duration += time.perf_counter() - _start_time
                    train_cache.update({step: x_train_transform})
                else:
                    x_train_transform = train_cache[step]

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits = model(x_train_transform, training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(y_batch_train, logits)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Update training metric.
                train_acc_metric.update_state(y_batch_train, logits)

                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * args["minibatch_size"]))

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for step, (x_batch_val, xx_batch_val, y_batch_val) in enumerate(val_dataset):
                if step not in val_cache.keys():
                    _start_time = time.perf_counter()
                    x_val_transform = transform(
                        x_batch_val, xx_batch_val,
                        base_parameters, diff1_parameters,
                        self.n_features_per_kernel
                    )
                    apply_kernel_on_train_duration += time.perf_counter() - _start_time
                    val_cache.update({step: x_val_transform})
                else:
                    x_val_transform = val_cache[step]

                val_logits = model(x_val_transform, training=False)
                # Update val metrics
                val_acc_metric.update_state(y_batch_val, val_logits)

            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))

        train_duration = time.perf_counter() - start_time

        if self.verbose > 1:
            print('[{}] Training done!, took {:.3f}s'.format(self.name, train_duration))

        # if predict_on_train:
        #     yhat = self.classifier.predict(x_train_transform)
        # else:
        #     yhat = None

        self.stats.update({
            "generate_kernel_duration": generate_kernel_duration,
            "train_transforms_duration": train_transforms_duration,
            "apply_kernel_on_train_duration": apply_kernel_on_train_duration,
            "train_duration": train_duration,
        })

        # return yhat

    def predict(self, x):
        if self.verbose > 1:
            print('[{}] Predicting'.format(self.name))

        if x.shape[2] < 10:
            # handling very short series (like PensDigit from the MTSC archive)
            # series have to be at least a length of 10 (including differencing)
            _x = np.zeros((x.shape[0], x.shape[1], 10), dtype=x.dtype)
            _x[:, :, :x.shape[2]] = x
            x = _x
            del _x

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

        return yhat, None
