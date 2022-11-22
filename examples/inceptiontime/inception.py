"""
    This is an adaption of the InceptionTime: https://github.com/hfawaz/InceptionTime
"""
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv1D,
    Concatenate,
    MaxPool1D,
    GlobalAveragePooling1D,
    Activation,
    Dropout,
    Layer,
    AveragePooling1D,
    SpatialDropout1D,
    Flatten,
    MaxPooling1D,
    LeakyReLU
)

from tensorflow.keras.regularizers import l2


class Inception:
    def __init__(
        self,
        use_bottleneck=True,
        bottleneck_size=32,
        kernel_size=40,
        nb_filters=32,
        depth=6,
        use_residual=True,
        increase_filters=False,
        increase_factor=2,
        l2_weight=0.0,
        dropout_rate=0.0,
        use_spatial_dropout=False,
        spatial_dropout_rate=0.1
    ):
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.depth = depth
        self.use_residual = use_residual
        self.increase_filters = increase_filters
        self.increase_factor = increase_factor
        self.l2_weight = l2_weight
        self.dropout_rate = dropout_rate
        self.use_spatial_dropout = use_spatial_dropout
        self.spatial_dropout_rate = spatial_dropout_rate

    def create_model(self, inputs):
        x = inputs
        input_res = inputs

        bottle_size = self.bottleneck_size
        nb_filters = self.nb_filters

        for d in range(self.depth):
            if self.increase_filters and d > 0 and d % 2 == 0:
                bottle_size *= self.increase_factor
                nb_filters *= self.increase_factor

            x = self._inception_module(
                x, bottleneck_size=bottle_size, nb_filters=nb_filters
            )
            if self.use_spatial_dropout:
                x = SpatialDropout1D(self.spatial_dropout_rate)(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = GlobalAveragePooling1D()(x)
        output = Dropout(self.dropout_rate)(gap_layer)

        return output

    def _inception_module(
        self, input_tensor, bottleneck_size, nb_filters, stride=1, activation="linear"
    ):
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(
                filters=bottleneck_size,
                kernel_size=1,
                padding="causal",
                activation=activation,
                use_bias=False,
                kernel_regularizer=l2(self.l2_weight),
            )(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(
                Conv1D(
                    filters=nb_filters,
                    kernel_size=kernel_size_s[i],
                    strides=stride,
                    padding="causal",
                    activation=activation,
                    use_bias=False,
                    kernel_regularizer=l2(self.l2_weight),
                )(input_inception)
            )

        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding="same")(
            input_tensor
        )

        conv_6 = Conv1D(
            filters=nb_filters,
            kernel_size=1,
            padding="causal",
            activation=activation,
            use_bias=False,
            kernel_regularizer=l2(self.l2_weight),
        )(max_pool_1)

        conv_list.append(conv_6)

        x = Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = Conv1D(
            filters=int(out_tensor.shape[-1]),
            kernel_size=1,
            padding="causal",
            use_bias=False,
            kernel_regularizer=l2(self.l2_weight),
        )(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)

        x = Add()([shortcut_y, out_tensor])
        x = LeakyReLU()(x)
        return x
