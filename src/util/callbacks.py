import tensorflow as tf


class CustomCheckpointCallback(tf.keras.callbacks.Callback):
    """A custom callback to save checkpoints. This callback calls the save function for this specific model."""

    def __init__(self, filepath, only_save_cpn=False, overwrite=True, verbose=False, **kwargs):
        """Constructor

        Args:
            filepath: The path where the checkpoints will be saved to
            overwrite: Whether files with the same name should be overwritten. Defaults to True.
            verbose: Print the saving progress. Defaults to False.
        """
        super().__init__()
        self.filepath = filepath
        self.overwrite = overwrite
        self.verbose = verbose
        self.only_save_cpn = only_save_cpn

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            filename = f"epoch_{epoch}"
            self.model.save(
                self.filepath, filename, self.only_save_cpn, self.overwrite, self.verbose
            )
