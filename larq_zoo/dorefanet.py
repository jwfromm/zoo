from typing import Optional, Tuple

import larq as lq
import tensorflow as tf
from tensorflow import keras
from zookeeper import component, task
from zookeeper.tf import Dataset, Model

from larq_zoo import utils
from larq_zoo.train import TrainLarqZooModel
from larq_zoo.weights import Weights


@lq.utils.register_keras_custom_object
@lq.utils.set_precision(1)
def magnitude_aware_sign_unclipped(x):
    """
    Scaled sign function with identity pseudo-gradient as used for the
    weights in the DoReFa paper. The Scale factor is calculated per layer.
    """
    scale_factor = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))

    @tf.custom_gradient
    def _magnitude_aware_sign(x):
        return lq.math.sign(x) * scale_factor, lambda dy: dy

    return _magnitude_aware_sign(x)


@lq.utils.register_keras_custom_object
def clip_by_value_activation(x):
    return tf.clip_by_value(x, 0, 1)


@component
class DoReFaNet(Model):
    """
    # References
    - [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low
    Bitwidth Gradients](https://arxiv.org/abs/1606.06160)
    """

    dataset: Dataset

    activations_k_bit: int = 2

    @property
    def input_quantizer(self):
        return lq.quantizers.DoReFaQuantizer(k_bit=self.activations_k_bit)

    @property
    def kernel_quantizer(self):
        return magnitude_aware_sign_unclipped

    def conv_block(
        self, x, filters, kernel_size, strides=1, pool=False, pool_padding="same"
    ):
        x = lq.layers.QuantConv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_constraint=None,
            use_bias=False,
        )(x)
        x = keras.layers.BatchNormalization(scale=False, momentum=0.9, epsilon=1e-4)(x)
        if pool:
            x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding=pool_padding)(x)
        return x

    def fully_connected_block(self, x, units):
        x = lq.layers.QuantDense(
            units,
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_constraint=None,
            use_bias=False,
        )(x)
        x = keras.layers.BatchNormalization(scale=False, momentum=0.9, epsilon=1e-4)(x)
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

        out = keras.layers.Conv2D(
            96, kernel_size=12, strides=4, padding="valid", use_bias=True
        )(img_input)
        out = self.conv_block(out, filters=256, kernel_size=5, pool=True)
        out = self.conv_block(out, filters=384, kernel_size=3, pool=True)
        out = self.conv_block(out, filters=384, kernel_size=3)
        out = self.conv_block(
            out, filters=256, kernel_size=3, pool_padding="valid", pool=True
        )

        if include_top:
            out = keras.layers.Flatten()(out)
            out = self.fully_connected_block(out, units=4096)
            out = self.fully_connected_block(out, units=4096)
            out = keras.layers.Activation("clip_by_value_activation")(out)
            out = keras.layers.Dense(self.dataset.num_classes, use_bias=True)(out)
            out = keras.layers.Activation("softmax")(out)

        model = keras.Model(inputs=img_input, outputs=out, name="dorefanet")

        # Load weights.
        if weights == "imagenet":
            # download appropriate file
            if include_top:
                weights_path = Weights(
                    model="dorefanet",
                    version="v0.1.0",
                    file="dorefanet_weights.h5",
                    file_hash="645d7839d574faa3eeeca28f3115773d75da3ab67ff6876b4de12d10245ecf6a",
                ).get_path()
            else:
                weights_path = Weights(
                    model="dorefanet",
                    version="v0.1.0",
                    file="dorefanet_weights_notop.h5",
                    file_hash="679368128e19a2a181bfe06ca3a3dec368b1fd8011d5f42647fbbf5a7f36d45f",
                ).get_path()
            model.load_weights(weights_path)
        elif weights is not None:
            model.load_weights(weights)
        return model


@task
class TrainDoReFaNet(TrainLarqZooModel):
    model = DoReFaNet()

    epochs = 90
    batch_size = 256

    learning_rate: float = 2e-4
    decay_start: int = 60
    decay_step_2: int = 75
    fast_decay_start: int = 82

    def learning_rate_schedule(self, epoch):
        if epoch < self.decay_start:
            return self.learning_rate
        elif epoch < self.decay_step_2:
            return self.learning_rate * 0.2
        elif epoch < self.fast_decay_start:
            return self.learning_rate * 0.2 * 0.2
        else:
            return (
                self.learning_rate
                * 0.2
                * 0.2
                * 0.1 ** ((epoch - self.fast_decay_start) // 2 + 1)
            )

    @property
    def optimizer(self):
        return keras.optimizers.Adam(self.learning_rate, epsilon=1e-5)
