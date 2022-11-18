import os
import pickle
import glob
from typing import Iterable, Union
from pathlib import Path
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from sklearn.model_selection import train_test_split, StratifiedKFold
from sktime.transformations.panel.rocket import MiniRocket, MiniRocketMultivariate
from classifier import train_classifier
import scipy.signal

BATCH_SIZE = 64
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

TRAIN_DATA_DIR = Path("/dataset/train/")  # Location of input test data
# TRAIN_DATA_DIR = Path("/home/dnhu0002/workspace/data/msg/train")  # Location of input test data


def open_npy(x):
    x = tf.io.read_file(x)
    x = tf.io.decode_raw(x, tf.float64)
    x = tf.reshape(x, (9996,))
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
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((({"filepath": x_data["filepath"], "patient": x_data["patient"]}), y_data))

    if oversampling:
        dataset_choices = []
        for i in range(2):
            dt = dataset.filter(lambda x, y: tf.equal(y, i)).map(
                lambda x, y: (open_npy(x["filepath"]), y),
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
            lambda x, y: (open_npy(x["filepath"]), y), num_parallel_calls=tf.data.AUTOTUNE
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

    kernel_num: int = 5000,
    max_dilations: int = 32,
    epochs: int = 100,
    batch_size=BATCH_SIZE,
) -> (tf.keras.models.Model, Union[MiniRocket, MiniRocketMultivariate]):
    print("Fitting minirocket")

    train_cache_file = "/tmp/train.cache"
    validation_cache_file = "/tmp/validation.cache"
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
            cache=True,
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
            cache=True,
        )
    )

    lr_model = train_classifier(
        class_num=2,
        dims=9996,
        train_data=train_dataset,
        train_steps=2 * int(X_train.shape[0] / batch_size),
        validation_data=validation_dataset,
        epochs=epochs,
    )
    return lr_model


def train_general_model(data_path, save_path, train_labels):
    patient_num = train_labels["patient"].nunique()
    patients = train_labels["patient"].unique().tolist()
    print(f"#Patients: {patient_num}")
    print(f"Patients: {patients}")
    train_labels["patient"] = train_labels["patient"].map(
        lambda x: patients.index(x)
    )

    X_train = []
    X_validation = []
    y_train = []
    y_validation = []
    for patient, group in train_labels.groupby("patient"):
        train_sessions = []
        val_sessions = []

        preictal_sessions = group[group["label"] == 1]["session"].unique()
        ictal_sessions = list(set(group[group["label"] == 0]["session"].unique()) - set(preictal_sessions))

        idx = int(len(preictal_sessions) * 0.8)
        train_sessions.extend(preictal_sessions[:idx])
        val_sessions.extend(preictal_sessions[idx:])

        idx = int(len(ictal_sessions) * 0.8)
        train_sessions.extend(ictal_sessions[:idx])
        val_sessions.extend(ictal_sessions[idx:])

        train_sessions = group[group["session"].isin(train_sessions)]
        val_sessions = group[group["session"].isin(val_sessions)]

        X_train.append(train_sessions[["filepath", "patient"]])
        X_validation.append(val_sessions[["filepath", "patient"]])
        y_train.append(train_sessions["label"])
        y_validation.append(val_sessions["label"])

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

    ###
    print("Training model")
    lr_model = train_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        max_dilations=64,
        kernel_num=10000,
        epochs=300,
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    lr_model.save(save_path)


def train_patient_specific(data_path, save_path, train_labels):
    for patient, group in train_labels.groupby("patient"):
        train_sessions = []
        val_sessions = []

        preictal_sessions = group[group["label"] == 1]["session"].unique()
        ictal_sessions = list(set(group[group["label"] == 0]["session"].unique()) - set(preictal_sessions))

        idx = int(len(preictal_sessions) * 0.8)
        train_sessions.extend(preictal_sessions[:idx])
        val_sessions.extend(preictal_sessions[idx:])

        idx = int(len(ictal_sessions) * 0.8)
        train_sessions.extend(ictal_sessions[:idx])
        val_sessions.extend(ictal_sessions[idx:])

        train_sessions = group[group["session"].isin(train_sessions)]
        val_sessions = group[group["session"].isin(val_sessions)]

        X_train = train_sessions[["filepath", "patient"]]
        X_validation = val_sessions[["filepath", "patient"]]
        y_train = train_sessions["label"]
        y_validation = val_sessions["label"]

        print("# samples in each class in the train set")
        print(np.unique(y_train, return_counts=True))

        # Count samples in each class
        print("# samples in each class in the CV set")
        print(np.unique(y_validation, return_counts=True))

        batch_size = np.min([BATCH_SIZE, y_train[y_train == 1].shape[0]])
        ###
        print("Training model")
        lr_model = train_model(
            X_train,
            y_train,
            X_validation,
            y_validation,
            max_dilations=32,
            kernel_num=5000,
            epochs=300,
            batch_size=batch_size,
        )
        patient_model_save_path = os.path.join(save_path, str(patient))
        if not os.path.exists(patient_model_save_path):
            os.makedirs(patient_model_save_path)

        lr_model.save(patient_model_save_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--data-path", type=str, default="/dataset/train/", required=False
    )
    arg_parser.add_argument(
        "--preprocessed-path", type=str, default="/dataset/preprocessed/", required=False
    )
    arg_parser.add_argument(
        "--train-label", type=str, default="/dataset/train_labels", required=False
    )
    arg_parser.add_argument(
        "--save-path", type=str, default="/trained_model", required=False
    )
    arg_parser.add_argument(
        "--training-mode",
        type=str,
        default="general",
        required=False,
        choices=["general", "patient-specific"],
    )
    args = arg_parser.parse_args()

    data_path = Path(args.data_path)
    save_path = Path(args.save_path)
    preprocessed_path = Path(args.preprocessed_path)
    training_mode = args.training_mode

    print(
        f"GPU available: {tf.test.is_gpu_available()}"
    )

    train_labels = pd.read_csv(args.train_label)
    train_labels["patient"] = train_labels["filepath"].map(lambda x: x.split("/")[0])
    train_labels["session"] = train_labels["filepath"].map(lambda x: x.split("/")[1])

    train_labels["filepath"] = train_labels["filepath"].map(
        lambda x: os.path.join(preprocessed_path, os.path.join(**x.split("/")[:-1]), x.split("/")[-1].split(".")[0] + ".bin")
    )

    if training_mode == "general":
        train_general_model(data_path, save_path, train_labels)
    else:
        train_patient_specific(data_path, save_path, train_labels)
