import functools
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Union

import click
import larq as lq
import tensorflow as tf
from tensorflow import keras
from zookeeper import cli
from zookeeper.tf import Experiment

from larq_zoo import utils


class TrainLarqZooModel(Experiment):
    # Save model checkpoints.
    use_model_checkpointing: bool = True

    # Log metrics to Tensorboard.
    use_tensorboard: bool = True

    # Use a per-batch progress bar (as opposed to per-epoch).
    use_progress_bar: bool = False

    # Where to store output.
    output_dir: Union[str, Path]

    @property
    def output_dir(self):
        return (
            Path.home()
            / "zookeeper-logs"
            / self.dataset.__class__.__name__
            / self.__class__.__name__
            / datetime.now().strftime("%Y%m%d_%H%M")
        )

    @property
    def model_path(self):
        return self.output_dir / "model"

    metrics = ["sparse_categorical_accuracy", "sparse_top_k_categorical_accuracy"]

    loss = "sparse_categorical_crossentropy"

    @property
    def callbacks(self):
        callbacks = []
        if self.use_model_checkpointing:
            callbacks.append(
                utils.ModelCheckpoint(
                    filepath=str(self.model_path), save_weights_only=True
                )
            )
        if self.learning_rate_schedule:
            callbacks.append(
                keras.callbacks.LearningRateScheduler(self.learning_rate_schedule)
            )
        if self.use_tensorboard:
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=self.output_dir, write_graph=False, profile_batch=0
                )
            )
        return callbacks

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)

        initial_epoch = utils.get_current_epoch(self.output_dir)

        train_data, num_train_examples = self.dataset.train(
            decoders=self.preprocessing.decoders
        )
        train_data = (
            train_data.cache()
            .shuffle(10 * self.batch_size)
            .repeat()
            .map(
                functools.partial(self.preprocessing, training=True),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .batch(self.batch_size)
            .prefetch(1)
        )

        validation_data, num_validation_examples = self.dataset.validation(
            decoders=self.preprocessing.decoders
        )
        validation_data = (
            validation_data.cache()
            .repeat()
            .map(self.preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(self.batch_size)
            .prefetch(1)
        )

        with utils.get_distribution_scope(self.batch_size):
            model = self.model.build(
                input_shape=self.preprocessing.input_shape,
                include_top=True,
                weights=None,
            )

            model.compile(
                optimizer=self.optimizer, loss=self.loss, metrics=self.metrics,
            )

            lq.models.summary(model)

            if initial_epoch > 0:
                model.load_weights(self.model_path)
                print(f"Loaded model from epoch {initial_epoch}.")

        click.secho(str(self))

        model.fit(
            train_data,
            epochs=self.epochs,
            steps_per_epoch=math.ceil(num_train_examples / self.batch_size),
            validation_data=validation_data,
            validation_steps=math.ceil(num_validation_examples / self.batch_size),
            verbose=1 if self.use_progress_bar else 2,
            initial_epoch=initial_epoch,
            callbacks=self.callbacks,
        )

        model_name = self.model.__class__.__name__

        # Save model and weights with top.
        model.save(self.output_dir / f"{model_name}.h5")
        model.save_weights(self.output_dir / f"{model_name}_weights.h5")

        # Save weights without top.
        notop_model = self.model.build(
            input_shape=self.preprocessing.input_shape, include_top=False, weights=None,
        )
        notop_model.set_weights(model.get_weights()[: len(notop_model.get_weights())])
        notop_model.save_weights(self.output_dir / f"{model_name}_weights_notop.h5")


if __name__ == "__main__":
    import importlib

    # Running it without the CLI requires us to first import larq_zoo
    # in order to register the models and datasets
    importlib.import_module("larq_zoo")
    cli()
