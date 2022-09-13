import glob
import os
from typing import Union, Iterable

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow_addons.optimizers import TriangularCyclicalLearningRate, SGDW

from tools import load_data


def tf_load_parquet(x) -> tf.Tensor:
    x = tf.py_function(load_data, [x], tf.float64)
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


class LogisticRegression:
    def __init__(
            self,
            num_features,
            max_epochs=500,
            minibatch_size=16,  # 256
            validation_size=2 ** 11,
            learning_rate=1e-4,
            patience_lr=5,  # 50 minibatches
            patience=10,  # 100 minibatches
            save_path=None,
    ):
        self.name = "LogisticRegression"

        self.args = {
            "num_features": num_features,
            "validation_size": validation_size,
            "minibatch_size": minibatch_size,
            "lr": learning_rate,
            "max_epochs": max_epochs,
            "patience_lr": patience_lr,
            "patience": patience,
        }
        self.model = None
        self.num_classes = None
        self.classes = None
        self.save_path = save_path

    def fit(self, x_train, y_train, positive_class=1):
        training_size = x_train.shape[0]

        args = self.args
        batch_size = np.min([args["minibatch_size"], y_train[y_train == positive_class].shape[0]])
        train_steps = int(training_size / batch_size)

        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x_train)

        self.classes = np.unique(y_train)
        self.num_classes = len(self.classes)

        self.enc = OneHotEncoder()
        if self.num_classes > 2:
            y_train = self.enc.fit_transform(y_train.reshape(-1, 1)).toarray()

        # -- model -----------------------------------------------------------------
        out_dims = self.num_classes if self.num_classes > 2 else 1
        out_activation = "softmax" if self.num_classes > 2 else "sigmoid"
        input_layer = tf.keras.layers.Input((x_train.shape[1],))
        output_layer = tf.keras.layers.Dense(
            out_dims,
            activation=out_activation,
            bias_initializer="zeros",
            kernel_initializer="zeros",
            kernel_regularizer=tf.keras.regularizers.l1_l2(1e-4, 1e-4)
        )(input_layer)
        model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer,
            name=self.name
        )
        model.summary()

        # lr scheduler
        lr_scheduler = TriangularCyclicalLearningRate(
            initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            step_size=8 * train_steps,
            # step_size=args["patience_lr"]
        )
        # Instantiate an optimizer
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
        optimizer = SGDW(learning_rate=lr_scheduler, weight_decay=1e-5)
        # Instantiate a loss function
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy() if self.num_classes > 2 else tf.keras.losses.BinaryCrossentropy()

        metrics = [
            tf.keras.metrics.AUC(
                curve="PR",
                num_thresholds=10000,
                name="auprc",
            ),
            tf.keras.metrics.PrecisionAtRecall(
                0.8,
                name="prec",
                num_thresholds=10000
            ),
            tf.keras.metrics.SpecificityAtSensitivity(
                0.8,
                name="specs",
                num_thresholds=10000
            ),
        ]

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )

        # -- validation data -------------------------------------------------------
        # args["validation_size"] = np.minimum(args["validation_size"], int(0.3 * training_size))
        args["validation_size"] = int(0.2 * training_size)
        if args["validation_size"] < training_size:
            x_training, x_validation, y_training, y_validation = train_test_split(
                x_train, y_train,
                test_size=args["validation_size"],
                stratify=y_train
            )

            train_history = model.fit(
                x_training, y_training,
                validation_data=(x_validation, y_validation),
                epochs=args["max_epochs"],
                steps_per_epoch=train_steps,
            )
        else:
            train_history = model.fit(
                x_train, y_train,
                epochs=args["max_epochs"],
                steps_per_epoch=train_steps,
            )
        self.model = model

    def fit_large(self, X_train, y_train,
                  X_validation, y_validation,
                  transform_func, dim,
                  positive_class=1):

        batch_size = np.min([self.args["minibatch_size"], y_train[y_train == 1].shape[0]])

        train_cache_file = self.save_path + "/train.cache"
        validation_cache_file = self.save_path + "/validation.cache"
        for f in glob.glob(train_cache_file + "*"):
            os.remove(f)
        for f in glob.glob(validation_cache_file + "*"):
            os.remove(f)

        self.classes = np.unique(y_train)
        self.num_classes = len(self.classes)

        self.enc = OneHotEncoder()
        if self.num_classes > 2:
            y_train = self.enc.fit_transform(y_train.reshape(-1, 1)).toarray()

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
                transform_func=transform_func,
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
                transform_func=transform_func,
            )
        )

        training_size = X_train.shape[0]

        args = self.args
        batch_size = np.min([args["minibatch_size"], y_train[y_train == positive_class].shape[0]])
        train_steps = 2 * int(training_size / batch_size)

        # -- model -----------------------------------------------------------------
        out_dims = self.num_classes if self.num_classes > 2 else 1
        out_activation = "softmax" if self.num_classes > 2 else "sigmoid"
        input_layer = tf.keras.layers.Input((dim,))
        input_layer = tf.keras.layers.Normalization(axis=-1)(input_layer)
        output_layer = tf.keras.layers.Dense(
            out_dims,
            activation=out_activation,
            bias_initializer="zeros",
            kernel_initializer="zeros",
            kernel_regularizer=tf.keras.regularizers.l1_l2(1e-4, 1e-4)
        )(input_layer)
        model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer,
            name=self.name
        )
        model.summary()

        # lr scheduler
        lr_scheduler = TriangularCyclicalLearningRate(
            initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            step_size=8 * train_steps,
            # step_size=args["patience_lr"]
        )
        # Instantiate an optimizer
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
        optimizer = SGDW(learning_rate=lr_scheduler, weight_decay=1e-5)
        # Instantiate a loss function
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy() if self.num_classes > 2 else tf.keras.losses.BinaryCrossentropy()

        metrics = [
            tf.keras.metrics.AUC(
                curve="PR",
                num_thresholds=10000,
                name="auprc",
            ),
            tf.keras.metrics.PrecisionAtRecall(
                0.8,
                name="prec",
                num_thresholds=10000
            ),
            tf.keras.metrics.SpecificityAtSensitivity(
                0.8,
                name="specs",
                num_thresholds=10000
            ),
        ]

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )

        train_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=args["max_epochs"],
            steps_per_epoch=train_steps,
        )
        self.model = model

    def predict(self, x):
        x = self.scaler.transform(x)

        yhat = self.model.predict(x)
        # if self.num_classes > 2:
        #     yhat = self.classes[np.argmax(yhat, axis=1)]
        # else:
        #     yhat = np.round(yhat)

        return yhat

    def predict_large(self, x, transform_func):
        test_dataset = (
            tf.data.Dataset.from_tensor_slices(x)
                .map(tf_load_parquet)
                .batch(32, drop_remainder=False)
                .map(transform_func)
        )

        yhat = self.model.predict(test_dataset)
        # if self.num_classes > 2:
        #     yhat = self.classes[np.argmax(yhat, axis=1)]
        # else:
        #     yhat = np.round(yhat)

        return yhat
