from typing import Callable, Optional, Tuple, Union

import larq as lq
import tensorflow as tf
from tensorflow import keras
from zookeeper import component, task
from zookeeper.tf import Dataset, Model

from larq_zoo import utils
from larq_zoo.train import TrainLarqZooModel
from larq_zoo.weights import Weights


@lq.utils.set_precision(1)
@lq.utils.register_keras_custom_object
def xnor_weight_scale(x):
    """ Clips the weights between -1 and +1 and then
        calculates a scale factor per weight filter. See
        https://arxiv.org/abs/1603.05279 for more details
    """

    x = tf.clip_by_value(x, -1, 1)

    alpha = tf.reduce_mean(tf.abs(x), axis=[0, 1, 2], keepdims=True)

    return alpha * lq.quantizers.ste_sign(x)


@component
class XNORNet(Model):
    """
    # References
    - [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural
      Networks](https://arxiv.org/abs/1603.05279)
    """

    dataset: Dataset

    use_bias: bool = False
    bn_scale: bool = False
    bn_momentum: float = 0.9

    input_quantizer: Union[Callable, str] = "ste_sign"
    kernel_quantizer: Union[Callable, str] = "xnor_weight_scale"
    kernel_constraint: Union[Callable, str] = "weight_clip"

    kernel_regularizer: Union[Callable, str]

    @property
    def kernel_regularizer(self):
        return keras.regularizers.l2(5e-7)

    def build(
        self,
        input_shape: Optional[Tuple[int, int, int]] = None,
        input_tensor: Optional[tf.Tensor] = None,
        include_top=True,
        weights="imagenet",
    ) -> keras.models.Model:

        kwargs = dict(
            kernel_quantizer=self.kernel_quantizer,
            input_quantizer=self.input_quantizer,
            kernel_constraint=self.kernel_constraint,
            use_bias=self.use_bias,
            kernel_regularizer=self.kernel_regularizer,
        )

        input_shape = utils.validate_input(
            input_shape, weights, include_top, self.dataset.num_classes
        )
        img_input = utils.get_input_layer(input_shape, input_tensor)

        x = keras.layers.Conv2D(
            96,
            (11, 11),
            strides=(4, 4),
            padding="same",
            use_bias=self.use_bias,
            input_shape=input_shape,
            kernel_regularizer=self.kernel_regularizer,
        )(img_input)

        x = keras.layers.BatchNormalization(
            momentum=self.bn_momentum, scale=self.bn_scale, epsilon=1e-5
        )(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = keras.layers.BatchNormalization(
            momentum=self.bn_momentum, scale=self.bn_scale, epsilon=1e-4
        )(x)
        x = lq.layers.QuantConv2D(256, (5, 5), padding="same", **kwargs)(x)
        x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = keras.layers.BatchNormalization(
            momentum=self.bn_momentum, scale=self.bn_scale, epsilon=1e-4
        )(x)
        x = lq.layers.QuantConv2D(384, (3, 3), padding="same", **kwargs)(x)
        x = keras.layers.BatchNormalization(
            momentum=self.bn_momentum, scale=self.bn_scale, epsilon=1e-4
        )(x)
        x = lq.layers.QuantConv2D(384, (3, 3), padding="same", **kwargs)(x)
        x = keras.layers.BatchNormalization(
            momentum=self.bn_momentum, scale=self.bn_scale, epsilon=1e-4
        )(x)
        x = lq.layers.QuantConv2D(256, (3, 3), padding="same", **kwargs)(x)
        x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = keras.layers.BatchNormalization(
            momentum=self.bn_momentum, scale=self.bn_scale, epsilon=1e-4
        )(x)
        x = lq.layers.QuantConv2D(4096, (6, 6), padding="valid", **kwargs)(x)
        x = keras.layers.BatchNormalization(
            momentum=self.bn_momentum, scale=self.bn_scale, epsilon=1e-4
        )(x)

        if include_top:
            # Equivilent to a dense layer
            x = lq.layers.QuantConv2D(
                4096, (1, 1), strides=(1, 1), padding="valid", **kwargs
            )(x)
            x = keras.layers.BatchNormalization(
                momentum=self.bn_momentum, scale=self.bn_scale, epsilon=1e-3
            )(x)
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(
                self.dataset.num_classes,
                use_bias=False,
                kernel_regularizer=self.kernel_regularizer,
            )(x)
            x = keras.layers.Activation("softmax")(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = keras.utils.get_source_inputs(input_tensor)
        else:
            inputs = img_input

        model = keras.models.Model(inputs, x, name="xnornet")

        # Load weights.
        if weights == "imagenet":
            # Download appropriate file
            if include_top:
                weights_path = Weights(
                    model="xnornet",
                    version="v0.2.0",
                    file="xnornet_weights.h5",
                    file_hash="e6ba24f785655260ae76a2ef1fab520e3528243d9c8fac430299cd81dbeabe10",
                ).get_path()
            else:
                weights_path = Weights(
                    model="xnornet",
                    version="v0.2.0",
                    file="xnornet_weights_notop.h5",
                    file_hash="0b8e3d0d60a7a728b5e387b8cd9f0fedc1dd72bcf9f4c693a2245d3a3c596b91",
                ).get_path()
            model.load_weights(weights_path)
        elif weights is not None:
            model.load_weights(weights)
        return model


@task
class TrainXNORNet(TrainLarqZooModel):
    model = XNORNet()

    epochs = 100
    batch_size = 1200

    initial_lr = 0.001

    def learning_rate_schedule(self, epoch):
        epoch_dec_1 = 19
        epoch_dec_2 = 30
        epoch_dec_3 = 44
        epoch_dec_4 = 53
        epoch_dec_5 = 66
        epoch_dec_6 = 76
        epoch_dec_7 = 86
        if epoch < epoch_dec_1:
            return self.initial_lr
        elif epoch < epoch_dec_2:
            return self.initial_lr * 0.5
        elif epoch < epoch_dec_3:
            return self.initial_lr * 0.1
        elif epoch < epoch_dec_4:
            return self.initial_lr * 0.1 * 0.5
        elif epoch < epoch_dec_5:
            return self.initial_lr * 0.01
        elif epoch < epoch_dec_6:
            return self.initial_lr * 0.01 * 0.5
        elif epoch < epoch_dec_7:
            return self.initial_lr * 0.01 * 0.1
        else:
            return self.initial_lr * 0.001 * 0.1

    @property
    def kernel_regularizer(self):
        return keras.regularizers.l2(self.regularization_quantity)

    @property
    def optimizer(self):
        return keras.optimizers.Adam(self.initial_lr)
