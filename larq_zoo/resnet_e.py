from typing import Optional, Tuple

import larq as lq
import tensorflow as tf
from tensorflow import keras
from zookeeper import component, task
from zookeeper.tf import Dataset, Model

from larq_zoo import utils
from larq_zoo.train import TrainLarqZooModel
from larq_zoo.weights import Weights


@component
class BinaryResNetE18(Model):
    """
    # References
    - [Back to Simplicity: How to Train Accurate BNNs from
      Scratch?](https://arxiv.org/abs/1906.08637)
    """

    dataset: Dataset

    num_layers: int = 18
    initial_filters: int = 64

    quantizer = lq.quantizers.SteSign(clip_value=1.25)
    constraint = lq.constraints.WeightClip(clip_value=1.25)

    @property
    def spec(self):
        spec = {
            18: ([2, 2, 2, 2], [64, 128, 256, 512]),
            34: ([3, 4, 6, 3], [64, 128, 256, 512]),
            50: ([3, 4, 6, 3], [256, 512, 1024, 2048]),
            101: ([3, 4, 23, 3], [256, 512, 1024, 2048]),
            152: ([3, 8, 36, 3], [256, 512, 1024, 2048]),
        }
        try:
            return spec[self.num_layers]
        except Exception:
            raise ValueError(f"Only specs for layers {list(self.spec.keys())} defined.")

    def residual_block(self, x, filters, strides=1):
        downsample = x.get_shape().as_list()[-1] != filters

        if downsample:
            residual = keras.layers.AvgPool2D(pool_size=2, strides=2)(x)
            residual = keras.layers.Conv2D(
                filters,
                kernel_size=1,
                use_bias=False,
                kernel_initializer="glorot_normal",
            )(residual)
            residual = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(
                residual
            )
        else:
            residual = x

        x = lq.layers.QuantConv2D(
            filters,
            kernel_size=3,
            strides=strides,
            padding="same",
            input_quantizer=self.quantizer,
            kernel_quantizer=self.quantizer,
            kernel_constraint=self.constraint,
            kernel_initializer="glorot_normal",
            use_bias=False,
            metrics=[],
        )(x)
        x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

        return keras.layers.add([x, residual])

    def build(
        self,
        input_shape: Optional[Tuple[int, int, int]] = None,
        input_tensor: Optional[tf.Tensor] = None,
        include_top=True,
        weights="imagenet",
    ) -> keras.models.Model:

        input_shape = utils.validate_input(
            input_shape, weights, include_top, self.dataset.num_classes
        )
        img_input = utils.get_input_layer(input_shape, input_tensor)

        if input_shape[0] and input_shape[0] < 50:
            x = keras.layers.Conv2D(
                self.initial_filters,
                kernel_size=3,
                padding="same",
                kernel_initializer="he_normal",
                use_bias=False,
            )(img_input)
        else:
            x = keras.layers.Conv2D(
                self.initial_filters,
                kernel_size=7,
                strides=2,
                padding="same",
                kernel_initializer="he_normal",
                use_bias=False,
            )(img_input)

            x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.MaxPool2D(3, strides=2, padding="same")(x)
            x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

        for block, (layers, filters) in enumerate(zip(*self.spec)):
            # This trick adds shortcut connections between original ResNet
            # blocks. We wultiply the number of blocks by two, but add only one
            # layer instead of two in each block
            for layer in range(layers * 2):
                strides = 1 if block == 0 or layer != 0 else 2
                x = self.residual_block(x, filters, strides=strides)

        if include_top:
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.GlobalAvgPool2D()(x)
            x = keras.layers.Dense(
                self.dataset.num_classes,
                activation="softmax",
                kernel_initializer="glorot_normal",
            )(x)

        model = keras.Model(
            inputs=img_input, outputs=x, name=f"binary_resnet_e_{self.num_layers}"
        )

        # Load weights.
        if weights == "imagenet":
            # Download appropriate file
            if include_top:
                weights_path = Weights(
                    model="resnet_e",
                    version="v0.1.0",
                    file="resnet_e_18_weights.h5",
                    file_hash="bde4a64d42c164a7b10a28debbe1ad5b287c499bc0247ecb00449e6e89f3bf5b",
                ).get_path()
            else:
                weights_path = Weights(
                    model="resnet_e",
                    version="v0.1.0",
                    file="resnet_e_18_weights_notop.h5",
                    file_hash="14cb037e47d223827a8d09db88ec73d60e4153a4464dca847e5ae1a155e7f525",
                ).get_path()
            model.load_weights(weights_path)
        elif weights is not None:
            model.load_weights(weights)
        return model


@task
class TrainBinaryResNetE18(TrainLarqZooModel):
    model = BinaryResNetE18()

    epochs = 120
    batch_size = 1024

    learning_rate = 0.004
    learning_factor = 0.3
    learning_steps = [70, 90, 110]

    def learning_rate_schedule(self, epoch):
        lr = self.learning_rate
        for step in self.learning_steps:
            if epoch < step:
                return lr
            lr *= self.learning_factor
        return lr

    @property
    def optimizer(self):
        return keras.optimizers.Adam(self.learning_rate, epsilon=1e-8)
