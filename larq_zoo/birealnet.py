from typing import Optional, Tuple

import larq as lq
import tensorflow as tf
from tensorflow import keras
from zookeeper import component
from zookeeper.tf import Dataset

from larq_zoo import utils
from larq_zoo.model import LarqZooModel


class BiRealNet(LarqZooModel):
    """
    # References
    - [Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved
      Representational Capability and Advanced Training
      Algorithm](https://arxiv.org/abs/1808.00278)
    """

    dataset: Dataset

    filters: int = 64

    input_quantizer: str = "approx_sign"
    kernel_quantizer: str = "magnitude_aware_sign"
    kernel_constraint: str = "weight_clip"
    kernel_initializer: str = "glorot_normal"

    @tf.function
    def residual_block(
        self, x, double_filters: bool = False, filters: Optional[int] = None
    ) -> tf.Tensor:
        assert not (double_filters and filters)

        # Compute dimensions
        in_filters = x.get_shape().as_list()[-1]
        out_filters = filters or in_filters if not double_filters else 2 * in_filters

        shortcut = x

        if in_filters != out_filters:
            shortcut = keras.layers.AvgPool2D(2, strides=2, padding="same")(shortcut)
            shortcut = keras.layers.Conv2D(
                out_filters,
                (1, 1),
                kernel_initializer=self.kernel_initializer,
                use_bias=False,
            )(shortcut)
            shortcut = keras.layers.BatchNormalization(momentum=0.8)(shortcut)

        x = lq.layers.QuantConv2D(
            out_filters,
            (3, 3),
            strides=1 if out_filters == in_filters else 2,
            padding="same",
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_initializer=self.kernel_initializer,
            kernel_constraint=self.kernel_constraint,
            use_bias=False,
        )(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)

        return keras.layers.add([x, shortcut])

    def build(
        self,
        input_shape: Optional[Tuple[int, int, int]] = None,
        input_tensor: Optional[tf.Tensor] = None,
    ) -> keras.models.Model:

        input_shape = utils.validate_input(
            input_shape, self.weights, self.include_top, self.dataset.num_classes
        )
        img_input = utils.get_input_layer(input_shape, input_tensor)

        # Layer 1
        out = keras.layers.Conv2D(
            self.filters,
            (7, 7),
            strides=2,
            kernel_initializer=self.kernel_initializer,
            padding="same",
            use_bias=False,
        )(img_input)
        out = keras.layers.BatchNormalization(momentum=0.8)(out)
        out = keras.layers.MaxPool2D((3, 3), strides=2, padding="same")(out)

        # Layer 2
        out = self.residual_block(out, filters=self.filters)

        # Layer 3 - 5
        for _ in range(3):
            out = self.residual_block(out)

        # Layer 6 - 17
        for _ in range(3):
            out = self.residual_block(out, double_filters=True)
            for _ in range(3):
                out = self.residual_block(out)

        # Layer 18
        if self.include_top:
            out = keras.layers.GlobalAvgPool2D()(out)
            out = keras.layers.Dense(self.dataset.num_classes, activation="softmax")(
                out
            )

        model = keras.Model(inputs=img_input, outputs=out, name="birealnet18")

        # Load weights.
        if self.weights == "imagenet":
            # Download appropriate file
            if self.include_top:
                weights_path = utils.download_pretrained_model(
                    model="birealnet",
                    version="v0.3.0",
                    file="birealnet_weights.h5",
                    file_hash="6e6efac1584fcd60dd024198c87f42eb53b5ec719a5ca1f527e1fe7e8b997117",
                )
            else:
                weights_path = utils.download_pretrained_model(
                    model="birealnet",
                    version="v0.3.0",
                    file="birealnet_weights_notop.h5",
                    file_hash="5148b61c0c2a1094bdef811f68bf4957d5ba5f83ad26437b7a4a6855441ab46b",
                )
            model.load_weights(weights_path)
        elif self.weights is not None:
            model.load_weights(self.weights)
        return model
