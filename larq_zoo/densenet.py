from typing import Callable, Optional, Sequence, Tuple, Union

import larq as lq
import tensorflow as tf
from tensorflow import keras
from zookeeper import component, task
from zookeeper.tf import Dataset, Model

from larq_zoo import utils
from larq_zoo.train import TrainLarqZooModel
from larq_zoo.weights import Weights


class BinaryDenseNet(Model):
    """
    # References
    - [Back to Simplicity: How to Train Accurate BNNs from
      Scratch?](https://arxiv.org/abs/1906.08637)
    """

    dataset: Dataset

    name: str

    initial_filters: int
    growth_rate: int
    reduction: Sequence[float]
    dilation_rate: Sequence[int]
    layers: Sequence[float]
    quantizer: Union[Callable, str]
    constraint = Union[Callable, str]

    imagenet_weights: Weights
    imagenet_no_top_weights: Weights

    def densely_connected_block(self, x, dilation_rate: int = 1):
        y = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        y = lq.layers.QuantConv2D(
            filters=self.growth_rate,
            kernel_size=3,
            dilation_rate=dilation_rate,
            input_quantizer=self.quantizer,
            kernel_quantizer=self.quantizer,
            kernel_initializer="glorot_normal",
            kernel_constraint=self.constraint,
            padding="same",
            use_bias=False,
        )(y)
        return keras.layers.concatenate([x, y])

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

        for block, layers_per_block in enumerate(self.layers):
            for _ in range(layers_per_block):
                x = self.densely_connected_block(x, self.dilation_rate[block])

            if block < len(self.layers) - 1:
                x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
                if self.dilation_rate[block + 1] == 1:
                    x = keras.layers.MaxPooling2D(2, strides=2)(x)
                x = keras.layers.Activation("relu")(x)
                x = keras.layers.Conv2D(
                    round(x.shape.as_list()[-1] // self.reduction[block] / 32) * 32,
                    kernel_size=1,
                    kernel_initializer="he_normal",
                    use_bias=False,
                )(x)

        x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

        if include_top:
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.GlobalAvgPool2D()(x)
            x = keras.layers.Dense(
                self.dataset.num_classes,
                activation="softmax",
                kernel_initializer="he_normal",
            )(x)

        model = keras.Model(inputs=img_input, outputs=x, name=self.name)

        if weights == "imagenet":
            if include_top:
                weights_path = self.imagenet_weights.get_path()
            else:
                weights_path = self.imagenet_no_top_weights.get_path()
            print(weights_path)
            model.load_weights(weights_path)
        elif weights is not None:
            model.load_weights(weights)

        return model


@component
class BinaryDenseNet28(BinaryDenseNet):
    name = "binary_densenet28"

    initial_filters = 64
    growth_rate = 64
    reduction = [2.7, 2.7, 2.2]
    dilation_rate = [1, 1, 1, 1]
    layers = [6, 6, 6, 5]
    quantizer = lq.quantizers.SteSign(clip_value=1.3)
    constraint = lq.constraints.WeightClip(clip_value=1.3)

    imagenet_weights = Weights(
        model="binary_densenet",
        version="v0.1.0",
        file="binary_densenet_28_weights.h5",
        file_hash="21fe3ca03eed244df9c41a2219876fcf03e73800932ec96a3e2a76af4747ac53",
    )
    imagenet_no_top_weights = Weights(
        model="binary_densenet",
        version="v0.1.0",
        file="binary_densenet_28_weights_notop.h5",
        file_hash="a376df1e41772c4427edd1856072b934a89bf293bf911438bf6f751a9b2a28f5",
    )


@component
class BinaryDenseNet37(BinaryDenseNet28):
    name = "binary_densenet37"

    reduction = [3.3, 3.3, 4]
    layers = [6, 8, 12, 6]

    imagenet_weights = Weights(
        model="binary_densenet",
        version="v0.1.0",
        file="binary_densenet_37_weights.h5",
        file_hash="8056a5d52c3ed86a934893987d09a06f59a5166aa9bddcaedb050f111d0a7d76",
    )
    imagenet_no_top_weights = Weights(
        model="binary_densenet",
        version="v0.1.0",
        file="binary_densenet_37_weights_notop.h5",
        file_hash="4e12bca9fd27580a5b833241c4eb35d6cc332878c406048e6ca8dbbc78d59175",
    )


@component
class BinaryDenseNet37Dilated(BinaryDenseNet37):
    name = "binary_densenet37_dilated"

    dilation_rate = [1, 1, 2, 4]

    epochs = 80
    batch_size = 256
    learning_steps = [60, 70]

    imagenet_weights = Weights(
        model="binary_densenet",
        version="v0.1.0",
        file="binary_densenet_37_dilated_weights.h5",
        file_hash="15c1bcd79b8dc22971382fbf79acf364a3f51049d0e584a11533e6fdbb7363d3",
    )
    imagenet_no_top_weights = Weights(
        model="binary_densenet",
        version="v0.1.0",
        file="binary_densenet_37_dilated_weights_notop.h5",
        file_hash="eaf3eac19fc90708f56a27435fb06d0e8aef40e6e0411ff7a8eefbe479226e4f",
    )


@component
class BinaryDenseNet45(BinaryDenseNet28):
    name = "binary_densenet45"

    reduction = [2.7, 3.3, 4]
    layers = [6, 12, 14, 8]

    epochs = 125
    batch_size = 384
    learning_rate = 0.008
    learning_steps = [80, 100]

    imagenet_weights = Weights(
        model="binary_densenet",
        version="v0.1.0",
        file="binary_densenet_45_weights.h5",
        file_hash="d00a0d26fbd2dba1bfba8c0306c770f3aeea5c370e99f963bb239bd916f72c37",
    )
    imagenet_no_top_weights = Weights(
        model="binary_densenet",
        version="v0.1.0",
        file="binary_densenet_45_weights_notop.h5",
        file_hash="e72d5cc6b0afe4612f8be7b1f9bb48a53ba2c8468b57bf1266d2900c99fd2adf",
    )


@task
class TrainBinaryDenseNet28(TrainLarqZooModel):
    model: BinaryDenseNet = BinaryDenseNet28()

    epochs = 120
    batch_size = 256

    learning_rate: float = 4e-3
    learning_factor: float = 0.1
    learning_steps: Sequence[int] = [100, 110]

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


@task
class TrainBinaryDenseNet37(TrainBinaryDenseNet28):
    model = BinaryDenseNet37()
    batch_size = 192


@task
class TrainBinaryDenseNet37Dilated(TrainBinaryDenseNet37):
    model = BinaryDenseNet37Dilated()
    epochs = 80
    batch_size = 256
    learning_steps = [60, 70]


@task
class TrainBinaryDenseNet45(TrainBinaryDenseNet28):
    model = BinaryDenseNet45()
    epochs = 125
    batch_size = 384
    learning_rate = 0.008
    learning_steps = [80, 100]
