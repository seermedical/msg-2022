import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import scipy.signal
from p_tqdm import p_map


SEED = 42


def transform_data(file):
    x = pd.read_parquet(file, engine="pyarrow")
    x = x.fillna(0)
    x = np.transpose(x.values.tolist())
    f, t, Zxx = scipy.signal.stft(
        x,
        fs=128,
        window="hann",
        nperseg=128 * 10,
        scaling="psd",
        return_onesided=True,
    )
    Zxx = Zxx[:, [i for i in range(0, f.shape[0] + 1, 5)]]
    x = np.swapaxes(Zxx, 0, 1)
    x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    x = x.real.astype(np.float64)

    return x


def preprocess(params) -> str:
    file, save_path = params
    x = transform_data(file)

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


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--data-path", type=str, default="/dataset/train/", required=False
    )

    arg_parser.add_argument(
        "--train-label",
        type=str,
        default="/dataset/train/train_labels.csv",
        required=False,
    )

    arg_parser.add_argument(
        "--preprocessed-path", type=str, default=None, required=True
    )

    args = arg_parser.parse_args()

    train_labels = pd.read_csv(args.train_label)
    train_labels["filepath"] = train_labels["filepath"].map(
        lambda x: os.path.join(args.data_path, x)
    )

    p_map(
        preprocess,
        zip(
            train_labels["filepath"],
            [args.preprocessed_path] * train_labels.shape[0],
        ),
    )
