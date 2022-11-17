import os
import pickle
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal
from p_tqdm import p_map
from sktime.transformations.panel.rocket import MiniRocketMultivariate


SEED = 42


def preprocess(params) -> str:
    if type(params) is str:
        file = params
        save_path = None
        minirocket = None
    if len(params) == 2:
        file, save_path = params
        minirocket = None
    elif len(params) == 3:
        file, save_path, minirocket = params
        save_path = None

    x = pd.read_parquet(file, engine="pyarrow")
    x = x.fillna(0)
    x = np.transpose(x.values.tolist())
    f, t, Zxx = scipy.signal.stft(x, fs=128, window="hann", nperseg=64 * 10)
    x = np.reshape(Zxx, (Zxx.shape[0] * Zxx.shape[1], Zxx.shape[2]))

    x = x.real.astype(np.float64)

    if minirocket is not None:
        x = np.expand_dims(x, axis=0)
        x = minirocket.transform(x)
        x = np.array(x.values.tolist()).flatten()

    if save_path is None:
        return x
    else:
        processed_file = os.path.join(
            save_path, "/".join(file.split("/")[-4:]).split(".")[0] + ".bin"
        )
        folder = os.path.dirname(processed_file)
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
        except Exception as e:
            pass

        x.tofile(processed_file)
        return processed_file


def train_rocket(
    x, kernel_num: int = 10000, max_dilations: int = 32
) -> (np.ndarray, MiniRocketMultivariate):
    rocket = MiniRocketMultivariate(
        kernel_num, max_dilations_per_kernel=max_dilations, random_state=SEED
    )
    X_train_transform = rocket.fit_transform(x)

    return X_train_transform, rocket


def build_minirocket(sampled_files, save_path):
    preprocessed_data = p_map(preprocess, sampled_files)

    minirocket = MiniRocketMultivariate(num_features=10000, max_dilations_per_kernel=32)
    minirocket.fit(np.array(preprocessed_data))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, "minirocket.dat"), "w+b") as f:
        pickle.dump(minirocket, f)

    return minirocket


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--data-path", type=str, default="/dataset/train/", required=False
    )

    arg_parser.add_argument("--save-path", type=str, required=True)

    arg_parser.add_argument(
        "--preprocessed-path", type=str, default=None, required=True
    )

    args = arg_parser.parse_args()

    train_labels = pd.read_csv(os.path.join(args.data_path, "train_labels.csv"))
    train_labels["filepath"] = train_labels["filepath"].map(
        lambda x: os.path.join(args.data_path, x)
    )

    pos_samples = train_labels[train_labels["label"] == 1]
    pos_samples = pos_samples.sample(np.min([100, pos_samples.shape[0]]))["filepath"]

    neg_samples = train_labels[train_labels["label"] == 0]
    neg_samples = neg_samples.sample(np.min([100, neg_samples.shape[0]]))["filepath"]

    sampled_data = np.concatenate([neg_samples, pos_samples])
    minirocket = build_minirocket(sampled_data, args.save_path)

    p_map(
        preprocess,
        zip(
            train_labels["filepath"],
            [args.preprocessed_path] * train_labels.shape[0],
            [minirocket] * train_labels.shape[0],
        ),
    )
