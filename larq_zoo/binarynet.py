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
class BinaryAlexNet(Model):
    """
    Implementation of ["Binarized Neural
    Networks"](https://papers.nips.cc/paper/6573-binarized-neural-networks) by
    Hubara et al., NIPS, 2016.
    """

    dataset: Dataset

    inflation_ratio: int = 1

    kwhparams = dict(
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
        use_bias=False,
    )

    def conv_block(
        self,
        x: tf.Tensor,
        features: int,
        kernel_size: Tuple[int, int],
        strides: int = 1,
        pool: bool = False,
        first_layer: bool = False,
        no_inflation: bool = False,
    ):
        x = lq.layers.QuantConv2D(
            features * (1 if no_inflation else self.inflation_ratio),
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            input_quantizer=None if first_layer else "ste_sign",
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=False,
        )(x)
        if pool:
            x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
        x = keras.layers.BatchNormalization(scale=False, momentum=0.9)(x)
        return x

    def dense_block(self, x: tf.Tensor, units: int) -> tf.Tensor:
        x = lq.layers.QuantDense(units, **self.kwhparams)(x)
        x = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)(x)
        return x

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

        # Feature extractor
        out = self.conv_block(
            img_input,
            features=64,
            kernel_size=11,
            strides=4,
            pool=True,
            first_layer=True,
        )
        out = self.conv_block(out, features=192, kernel_size=5, pool=True)
        out = self.conv_block(out, features=384, kernel_size=3)
        out = self.conv_block(out, features=384, kernel_size=3)
        out = self.conv_block(
            out, features=256, kernel_size=3, pool=True, no_inflation=True
        )

        # Classifier
        if include_top:
            out = keras.layers.Flatten()(out)
            out = self.dense_block(out, units=4096)
            out = self.dense_block(out, units=4096)
            out = self.dense_block(out, self.dataset.num_classes)
            out = keras.layers.Activation("softmax")(out)

        model = keras.Model(inputs=img_input, outputs=out, name="binary_alexnet")

        # Load weights.
        if weights == "imagenet":
            # Download appropriate file
            if include_top:
                weights_path = Weights(
                    model="binary_alexnet",
                    version="v0.2.0",
                    file="binary_alexnet_weights.h5",
                    file_hash="0f8d3f6c1073ef993e2e99a38f8e661e5efe385085b2a84b43a7f2af8500a3d3",
                ).get_path()
            else:
                weights_path = Weights(
                    model="binary_alexnet",
                    version="v0.2.0",
                    file="binary_alexnet_weights_notop.h5",
                    file_hash="1c7e2ef156edd8e7615e75a3b8929f9025279a948d1911824c2f5a798042475e",
                ).get_path()
            model.load_weights(weights_path)
        elif weights is not None:
            model.load_weights(weights)

        return model


@task
class TrainBinaryNet(TrainLarqZooModel):
    model = BinaryAlexNet()

    batch_size = 512
    epochs = 150

    def learning_rate_schedule(self, epoch):
        return 1e-2 * 0.5 ** (epoch // 10)

    @property
    def optimizer(self):
        return keras.optimizers.Adam(self.learning_rate_schedule(0))
