import numpy as np
import pandas as pd
import psutil
from p_tqdm import p_map
from scipy import signal


def read_data(filepath, y=None):
    X = pd.read_parquet(filepath)

    X = X.fillna(0)
    X = X.transpose()
    X = X.values

    # resample
    X = signal.resample_poly(X, up=64, down=128, axis=-1)
    X = signal.resample_poly(X, up=64, down=128, axis=-1)
    f, t, Zxx = signal.stft(X, fs=64, window="hann", nperseg=64 * 10)
    Zxx = np.abs(Zxx)
    X = np.reshape(Zxx, (Zxx.shape[0] * Zxx.shape[1], Zxx.shape[2]))

    if y is None:
        return X, filepath
    return X, y, filepath


def sample(n, train_files, y):
    pos_y = y[y == 1]
    neg_y = y[y == 0]
    pos_samples = train_files[y == 1]
    neg_samples = train_files[y == 0]
    randint = np.random.permutation(neg_samples.shape[0])
    randint = randint[:n]
    neg_samples = neg_samples[randint]
    neg_y = neg_y[randint]

    sample_y = np.concatenate((pos_y, neg_y))
    sample_files = np.concatenate((pos_samples, neg_samples))
    return sample_files, sample_y


def load_parquets(files, y=None, n_sample=-1, num_cpus=-1):
    if num_cpus < 0:
        num_cpus = psutil.cpu_count(logical=True)
    else:
        num_cpus = min(num_cpus, psutil.cpu_count(logical=True))

    n_files = len(files)

    print(f"Loading {n_files} files.")
    if y is not None:
        print(f"label 0: {len(y[y == 0])}")
        print(f"label 1: {len(y[y == 1])}")

        # SAMPLE
        if n_sample > 0:
            sample_files, sample_y = sample(n_sample, files, y)
            print(f"label 0: {len(sample_y[sample_y == 0])}")
            print(f"label 1: {len(sample_y[sample_y == 0])}")

            tmp = p_map(read_data, sample_files, sample_y, num_cpus=num_cpus)
        else:
            tmp = p_map(read_data, files, y, num_cpus=num_cpus)

        f = []
        x = []
        y = []
        for i in tmp:
            x.append(i[0])
            y.append(i[1])
            f.append(i[1])
        del tmp

        x = np.array(x)
        y = np.array(y)

        return x, y, f
    else:
        # SAMPLE
        if n_sample > 0:
            randint = np.random.permutation(files.shape[0])
            randint = randint[:n_sample]
            sample_files = files[randint]

            tmp = p_map(read_data, sample_files, num_cpus=num_cpus)
        else:
            tmp = p_map(read_data, files, num_cpus=num_cpus)

        f = []
        x = []
        for i in tmp:
            x.append(i[0])
            f.append(i[1])
        del tmp

        x = np.array(x)

        return x, f
