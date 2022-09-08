import glob
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd
import scipy.signal
import tensorflow as tf
from sklearn.model_selection import train_test_split

from examples.multirocket.multirocket import train_multirocket, train_classifier, MultiRocket, MultiRocketMultivariate

BATCH_SIZE = 256 # 16
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

TRAIN_DATA_DIR = Path("./dataset/train/")  # Location of input train data
# TRAIN_DATA_DIR = Path("D:/Dataset/msg_contest_data/train")  # Location of input train data


def load_parquet(x) -> np.ndarray:
    if type(x) is not str:
        x = x.numpy().decode("utf-8")

    x = pd.read_parquet(x).iloc[:76800]
    x = x.fillna(0)
    x = np.transpose(x.values.tolist())
    # x = scipy.signal.resample(x, 128)
    x = scipy.signal.resample_poly(x, up=64, down=128, axis=-1)
    f, t, Zxx = scipy.signal.stft(x, fs=64, window="hann", nperseg=64 * 10)
    x = np.reshape(Zxx, (Zxx.shape[0] * Zxx.shape[1], Zxx.shape[2]))
    return x.real


def tf_load_parquet(x) -> tf.Tensor:
    x = tf.py_function(load_parquet, [x], tf.float64)
    return x


def create_dataset(
        x_data: Iterable[str],
        y_data: Iterable[int],
        batch_size: int = 32,
        drop_remainder: bool = False,
        oversampling: bool = False,
        shuffle: bool = False,
        shuffle_size: int = 1000,
        repeat: bool = False,
        cache: Union[bool, str] = False,
        transform_func=None,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

    if oversampling:
        dataset_choices = []
        for i in range(2):
            dt = dataset.filter(lambda x, y: tf.equal(y, i)).map(
                lambda x, y: (tf_load_parquet(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

            if transform_func:
                dt = dt.map(
                    transform_func,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )

            if cache:
                if type(cache) is str:
                    dt = dt.cache(f"{cache}_{i}")
                else:
                    dt = dt.cache()

            if shuffle:
                dt = dt.shuffle(shuffle_size)

            if repeat:
                dt = dt.repeat()
            dataset_choices.append(dt)

        dataset = tf.data.Dataset.sample_from_datasets(dataset_choices, [0.5, 0.5])
    else:
        dataset = dataset.map(
            lambda x, y: (tf_load_parquet(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        if transform_func:
            dataset = dataset.map(
                transform_func,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        if cache:
            if type(cache) is str:
                dataset = dataset.cache(cache)
            else:
                dataset = dataset.cache()

        if shuffle:
            dataset = dataset.shuffle(shuffle_size)

        if repeat:
            dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(
        tf.data.AUTOTUNE
    )

    data_options = tf.data.Options()
    # data_options.threading.max_intra_op_parallelism = 1
    data_options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    dataset = dataset.with_options(data_options)

    return dataset


def train_model(
        X_train: Union[pd.Series, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_validation: Union[pd.Series, np.ndarray],
        y_validation: Union[pd.Series, np.ndarray],
        num_features: int = 50_000,
        max_dilations: int = 32,
        epochs: int = 100,
        batch_size=BATCH_SIZE,
        save_path="",
) -> (tf.keras.models.Model, Union[MultiRocket, MultiRocketMultivariate]):
    print("Fitting multirocket")
    pos_samples = X_train[y_train == 1]
    pos_samples = pos_samples.sample(np.min([100, pos_samples.shape[0]]))

    neg_samples = X_train[y_train == 0]
    neg_samples = neg_samples.sample(np.min([100, neg_samples.shape[0]]))

    pos_samples = np.array([load_parquet(i) for i in pos_samples])
    neg_samples = np.array([load_parquet(i) for i in neg_samples])

    sampled_data = np.vstack([pos_samples, neg_samples])

    transformed_data, multirocket = train_multirocket(
        sampled_data, num_features=num_features, max_dilations=max_dilations
    )

    del sampled_data

    def multirocket_transform(x):
        x = x.numpy()
        x_shape = x.shape
        if len(x_shape) == 2:
            x = np.reshape(x, (1, x.shape[0], x.shape[1]))
        transformed_x = multirocket.transform(x)

        if len(x_shape) == 2:
            transformed_x = transformed_x.values.flatten()

        return transformed_x

    def tf_multirocket_transform(x, y):
        x = tf.py_function(multirocket_transform, [x], tf.float64)
        return x, y

    train_cache_file = save_path + "/train.cache"
    validation_cache_file = save_path + "/validation.cache"
    for f in glob.glob(train_cache_file + "*"):
        os.remove(f)

    for f in glob.glob(validation_cache_file + "*"):
        os.remove(f)

    train_dataset = (
        create_dataset(
            X_train,
            y_train,
            batch_size=batch_size,
            drop_remainder=True,
            shuffle=True,
            shuffle_size=100,
            repeat=True,
            oversampling=True,
            cache=train_cache_file,
            transform_func=tf_multirocket_transform,
        )
    )

    validation_dataset = (
        create_dataset(
            X_validation,
            y_validation,
            batch_size=batch_size,
            drop_remainder=False,
            shuffle=False,
            repeat=False,
            cache=validation_cache_file,
            transform_func=tf_multirocket_transform,
        )
    )

    lr_model = train_classifier(
        class_num=2,
        dims=transformed_data.shape[-1],
        train_data=train_dataset,
        train_steps=2 * int(X_train.shape[0] / batch_size),
        validation_data=validation_dataset,
        epochs=epochs,
        save_path=save_path
    )
    return lr_model, multirocket


def train_general_model(data_path, save_path):
    train_labels = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    train_labels["patient"] = train_labels["filepath"].map(lambda x: x.split("/")[0])
    train_labels["filepath"] = train_labels["filepath"].map(
        lambda x: os.path.join(data_path, x)
    )

    # SAMPLING DATA FOR DEMO
    sampled_data = []
    for patient, group in train_labels.groupby("patient"):
        pos_samples = group[group["label"] == 1]

        neg_samples = group[group["label"] == 0]
        neg_samples = neg_samples.sample(
            int(neg_samples.shape[0] / 2), random_state=SEED
        )

        sampled_data.extend([pos_samples, neg_samples])

    sampled_data = pd.concat(sampled_data)

    X_train = []
    X_validation = []
    y_train = []
    y_validation = []
    for patient, group in sampled_data.groupby("patient"):
        _x_train, _x_validation, _y_train, _y_validation = train_test_split(
            group["filepath"],
            group["label"],
            test_size=0.2,
            random_state=SEED,
        )
        X_train.append(_x_train)
        X_validation.append(_x_validation)
        y_train.append(_y_train)
        y_validation.append(_y_validation)

    X_train = pd.concat(X_train)
    X_validation = pd.concat(X_validation)
    y_train = pd.concat(y_train)
    y_validation = pd.concat(y_validation)

    # Count samples in each class
    print("# samples in each class in the train set")
    print(np.unique(y_train, return_counts=True))

    # Count samples in each class
    print("# samples in each class in the CV set")
    print(np.unique(y_validation, return_counts=True))

    batch_size = np.min([BATCH_SIZE, y_train[y_train == 1].shape[0]])
    # batch_size = BATCH_SIZE

    ###
    print("Training model")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    lr_model, multirocket = train_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        max_dilations=32,
        num_features=50_000,
        epochs=300,
        save_path=save_path,
        batch_size=batch_size
    )

    lr_model.save(save_path)

    with open(os.path.join(save_path, "multirocket.pkl"), "w+b") as f:
        pickle.dump(multirocket, f)


def train_patient_specific(data_path, save_path):
    train_labels = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    train_labels["patient"] = train_labels["filepath"].map(lambda x: x.split("/")[0])
    train_labels["filepath"] = train_labels["filepath"].map(
        lambda x: os.path.join(data_path, x)
    )

    # SAMPLING DATA FOR DEMO
    sampled_data = []
    for patient, group in train_labels.groupby("patient"):
        pos_samples = group[group["label"] == 1]

        neg_samples = group[group["label"] == 0]
        neg_samples = neg_samples.sample(
            int(neg_samples.shape[0] / 2), random_state=SEED
        )

        sampled_data.extend([pos_samples, neg_samples])

    sampled_data = pd.concat(sampled_data)

    for patient, group in sampled_data.groupby("patient"):
        X_train, X_validation, y_train, y_validation = train_test_split(
            group["filepath"],
            group["label"],
            test_size=0.2,
            random_state=SEED,
        )
        # Count samples in each class
        print("# samples in each class in the train set")
        print(np.unique(y_train, return_counts=True))

        # Count samples in each class
        print("# samples in each class in the CV set")
        print(np.unique(y_validation, return_counts=True))

        batch_size = np.min([BATCH_SIZE, y_train[y_train == 1].shape[0]])
        # batch_size = BATCH_SIZE
        ###
        print("Training model")
        patient_model_save_path = os.path.join(save_path, str(patient))
        if not os.path.exists(patient_model_save_path):
            os.makedirs(patient_model_save_path)

        lr_model, multirocket = train_model(
            X_train,
            y_train,
            X_validation,
            y_validation,
            max_dilations=32,
            num_features=50_000,
            epochs=300,
            batch_size=batch_size,
            save_path=patient_model_save_path,
        )

        lr_model.save(patient_model_save_path)

        with open(os.path.join(patient_model_save_path, "multirocket.pkl"), "w+b") as f:
            pickle.dump(multirocket, f)


MAIN_HERE = True
if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--data-path", type=str, default=TRAIN_DATA_DIR, required=False
    )
    arg_parser.add_argument(
        "--save-path", type=str, default="./trained_model", required=False
    )
    arg_parser.add_argument(
        "--training-mode",
        type=str,
        default="patient-specific",
        required=False,
        choices=["general", "patient-specific"],
    )
    args = arg_parser.parse_args()

    data_path = Path(args.data_path)
    save_path = Path(args.save_path)
    training_mode = args.training_mode

    print(
        f"GPU available:   {tf.test.is_gpu_available()}"
    )

    if training_mode == "general":
        train_general_model(data_path, save_path)
    else:
        train_patient_specific(data_path, save_path)
