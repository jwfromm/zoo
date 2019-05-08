from larq_flock import cli, build_train
import click


@cli.command()
@click.option("--tensorboard/--no-tensorboard", default=True)
@build_train
def train(build_model, dataset, hparams, output_dir, epochs, tensorboard):
    import larq as lq
    from larq_zoo.utils import get_distribution_scope
    import tensorflow as tf

    callbacks = [lq.callbacks.QuantizationLogger(update_freq=1000)]
    if tensorboard:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=output_dir,
                write_graph=False,
                update_freq=1000,
                histogram_freq=10,
            )
        )

    with get_distribution_scope(hparams.batch_size):
        model = build_model(hparams, dataset)
        model.compile(
            optimizer=hparams.optimizer(hparams.learning_rate),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy", "top_k_categorical_accuracy"],
        )

    lq.models.summary(model)

    model.fit(
        dataset.train_data(hparams.batch_size),
        epochs=epochs,
        steps_per_epoch=dataset.train_examples // hparams.batch_size,
        validation_data=dataset.validation_data(hparams.batch_size),
        validation_steps=dataset.validation_examples // hparams.batch_size,
        verbose=2 if tensorboard else 1,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    import importlib

    # Running it without the CLI requires us to first import larq_zoo
    # in order to register the models and datasets
    importlib.import_module("larq_zoo")
    cli()