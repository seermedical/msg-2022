import os
import pickle
import glob
from typing import Iterable, Union
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sktime.transformations.panel.rocket import MiniRocket, MiniRocketMultivariate
from minirocket import train_rocket, train_classifier
import scipy.signal

BATCH_SIZE = 128
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

TRAIN_DATA_DIR = Path("/dataset/train/")  # Location of input test data
# TRAIN_DATA_DIR = Path("/home/dnhu0002/workspace/data/msg/train")  # Location of input test data


def load_parquet(x) -> np.ndarray:
    if type(x) is not str:
        x = x.numpy().decode("utf-8")

    x = pd.read_parquet(x).iloc[:76800]
    x = x.fillna(0)
    x = np.transpose(x.values.tolist())
    x = scipy.signal.resample(x, num=768, axis=-1)
    return x


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
    X_train: Iterable,
    y_train: Iterable,
    X_validation: Iterable,
    y_validation: Iterable,
    kernel_num: int = 5000,
    max_dilations: int = 32,
    epochs: int = 100,
) -> (tf.keras.models.Model, Union[MiniRocket, MiniRocketMultivariate]):
    pos_samples = X_train[y_train == 1]
    pos_samples = pos_samples.sample(np.min([100, pos_samples.shape[0]]))

    neg_samples = X_train[y_train == 0]
    neg_samples = neg_samples.sample(np.min([100, neg_samples.shape[0]]))

    pos_samples = np.array(pos_samples.map(load_parquet).values.tolist())
    neg_samples = np.array(neg_samples.map(load_parquet).values.tolist())

    sampled_data = np.vstack([pos_samples, neg_samples])

    transformed_data, minirocket = train_rocket(
        sampled_data, kernel_num=kernel_num, max_dilations=max_dilations
    )

    del sampled_data

    def minirocket_transform(x):
        x = x.numpy()
        x_shape = x.shape
        if len(x_shape) == 2:
            x = np.reshape(x, (1, x.shape[0], x.shape[1]))
        transformed_x = minirocket.transform(x)

        if len(x_shape) == 2:
            transformed_x = transformed_x.values.flatten()

        return transformed_x

    def tf_minirocket_transform(x, y):
        x = tf.py_function(minirocket_transform, [x], tf.float64)
        return x, y

    train_cache_file = "train.cache"
    validation_cache_file = "validation.cache"
    for f in glob.glob(train_cache_file + "*"):
        os.remove(f)

    for f in glob.glob(validation_cache_file + "*"):
        os.remove(f)

    train_dataset = (
        create_dataset(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            drop_remainder=True,
            shuffle=True,
            shuffle_size=100,
            repeat=True,
            oversampling=True,
            cache=train_cache_file,
            transform_func=tf_minirocket_transform,
        )
        # .map(tf_minirocket_transform, num_parallel_calls=tf.data.AUTOTUNE)
        # .prefetch(tf.data.AUTOTUNE)
    )

    validation_dataset = (
        create_dataset(
            X_validation,
            y_validation,
            batch_size=BATCH_SIZE,
            drop_remainder=False,
            shuffle=False,
            repeat=False,
            cache=validation_cache_file,
            transform_func=tf_minirocket_transform,
        )
        # .map(tf_minirocket_transform, num_parallel_calls=tf.data.AUTOTUNE)
        # .prefetch(tf.data.AUTOTUNE)
    )

    lr_model = train_classifier(
        class_num=2,
        dims=transformed_data.shape[-1],
        train_data=train_dataset,
        train_steps=2 * int(X_train.shape[0] / BATCH_SIZE),
        validation_data=validation_dataset,
        save_path="trained_model",
        epochs=epochs,
    )
    return lr_model, minirocket


if __name__ == "__main__":
    print(
        f"GPU available:   {tf.test.is_gpu_available()}"
    )  # Use this if using tensorflow
    train_labels = pd.read_csv(os.path.join(TRAIN_DATA_DIR, "train_labels.csv"))
    train_labels["patient"] = train_labels["filepath"].map(lambda x: x.split("/")[0])
    train_labels["filepath"] = train_labels["filepath"].map(
        lambda x: os.path.join(TRAIN_DATA_DIR, x)
    )

    # SAMPLING DATA FOR DEMO
    # train_labels["patient"] = train_labels["filepath"].map(lambda x: x.split("/")[0])
    # sampled_data = []
    # for patient, group in train_labels.groupby("patient"):
    #     pos_samples = group[group["label"] == 1]
    #     pos_samples = pos_samples.sample(np.min([100, pos_samples.shape[0]]))
    #
    #     neg_samples = group[group["label"] == 0].sample(100)
    #     neg_samples = neg_samples.sample(np.min([100, neg_samples.shape[0]]))
    #
    #     sampled_data.extend([pos_samples, neg_samples])
    #
    # sampled_data = pd.concat(sampled_data)
    X_train = []
    X_validation = []
    y_train = []
    y_validation = []
    for patient, group in train_labels.groupby("patient"):
        _x_train, _x_validation, _y_train, _y_validation = train_test_split(
            group["filepath"],
            group["label"],
            test_size=0.2,
            random_state=SEED,
        )
        X_train.append(_x_train)
        X_validation.append(_y_validation)
        y_train.append(_y_train)
        y_validation.append(_y_validation)

    X_train = pd.concat(X_train)
    X_validation = pd.concat(X_validation)
    y_train = pd.concat(y_train)
    y_validation = pd.concat(y_validation)
    ###
    print("Training model")
    lr_model, minirocket = train_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        max_dilations=128,
        kernel_num=5000,
        epochs=300,
    )
    save_path = "/trained_model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    lr_model.save(save_path)

    with open(os.path.join(save_path, "minirocket.pkl"), "w+b") as f:
        pickle.dump(minirocket, f)
