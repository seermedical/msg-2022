# coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import (
    AUC,
    PrecisionAtRecall,
    RecallAtPrecision,
    SpecificityAtSensitivity,
)
from tensorflow_addons.optimizers import SGDW, TriangularCyclicalLearningRate
from tensorflow.keras.callbacks import EarlyStopping
from inception import Inception

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


class Classifier:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=self.input_shape)
        inception = Inception(
            use_spatial_dropout=True,
            spatial_dropout_rate=0.1,
            kernel_size=32,
            nb_filters=64,
            depth=8,
            use_residual=True,
            use_bottleneck=True,
        ).create_model(inputs)
        x = Dense(1, "sigmoid")(inception)
        model = Model(inputs, x)
        return model

    def train(
        self,
        train_data: tf.data.Dataset,
        validation_data: tf.data.Dataset,
        train_steps: int,
        epochs: int = 300,
    ):
        lr_scheduler = TriangularCyclicalLearningRate(
            initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            step_size=8 * train_steps,
        )
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
            PrecisionAtRecall(0.8, name="prec", num_thresholds=10000),
            SpecificityAtSensitivity(0.8, name="specs", num_thresholds=10000),
        ]
        monitor = "val_auc"

        self.model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=metrics
        )

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

        self.model.fit(
            train_data,
            validation_data=validation_data,
            steps_per_epoch=train_steps,
            epochs=epochs,
            callbacks=callbacks,
        )
