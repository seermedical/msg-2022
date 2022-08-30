# coding=utf-8
import pickle
import numpy as np
from sktime.transformations.panel.rocket import MiniRocket, MiniRocketMultivariate
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import (
    AUC,
    PrecisionAtRecall,
    RecallAtPrecision,
    SpecificityAtSensitivity,
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow_addons.optimizers import SGDW, TriangularCyclicalLearningRate
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from typing import Literal

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


def train_rocket(x, kernel_num=10000, max_dilations=32):
    if x.shape[1] == 1:
        rocket = MiniRocket(
            kernel_num, max_dilations_per_kernel=max_dilations, random_state=SEED
        )
    else:
        rocket = MiniRocketMultivariate(
            kernel_num, max_dilations_per_kernel=max_dilations, random_state=SEED
        )
    X_train_transform = rocket.fit_transform(x)

    return X_train_transform, rocket


def build_logistic_regression_model(dims: int, class_num=6):
    out_dims = class_num if class_num > 2 else 1
    inputs = Input(batch_shape=(None, dims))
    out_activation = "softmax" if out_dims > 1 else "sigmoid"
    x = Dense(
        out_dims, activation=out_activation, kernel_regularizer=l1_l2(1e-4, 1e-4)
    )(inputs)

    model = Model(inputs=inputs, outputs=x)

    model.summary()

    return model


def train_classifier(
    class_num: int,
    train_data: tf.data.Dataset,
    validation_data: tf.data.Dataset,
    train_steps: int,
    dims: int = 10000,
    epochs: int = 300,
    save_path: str = "",
):
    lr_scheduler = TriangularCyclicalLearningRate(
        initial_learning_rate=1e-4,
        maximal_learning_rate=1e-2,
        step_size=8 * train_steps,
    )
    loss = SparseCategoricalCrossentropy() if class_num > 2 else BinaryCrossentropy()
    model = build_logistic_regression_model(dims, class_num)

    optimizer = SGDW(learning_rate=lr_scheduler, weight_decay=1e-5)

    metrics = [
        AUC(
            num_thresholds=10000,
            name="auc",
        ),
        AUC(
            curve="PR",
            num_thresholds=10000,
            name="auprc",
        ),
        RecallAtPrecision(
            num_thresholds=10000,
            precision=0.8,
            name="sens",
        ),
        PrecisionAtRecall(0.8, name="prec", num_thresholds=10000),
        SpecificityAtSensitivity(0.8, name="specs", num_thresholds=10000),
    ]
    monitor = "val_auprc"

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = [
        EarlyStopping(
            monitor=monitor,
            mode="max",
            min_delta=1e-4,
            patience=15,
            verbose=1,
            restore_best_weights=True,
        ),
    ]

    train_history = model.fit(
        train_data,
        validation_data=validation_data,
        steps_per_epoch=train_steps,
        epochs=epochs,
        callbacks=callbacks,
    )
    model.save_weights(f"{save_path}/model_weights.hf5")

    with open(f"{save_path}/train_history.dat", "w+b") as f:
        pickle.dump(train_history.history, f)

    return model
