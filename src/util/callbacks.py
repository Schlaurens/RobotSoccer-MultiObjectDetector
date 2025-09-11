import tensorflow as tf


class CustomCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath, timestamp, **kwargs):
        super().__init__()
        self.filepath = filepath
        self.timestamp = timestamp

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.filepath, self.timestamp)
