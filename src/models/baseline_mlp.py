# coding: utf-8
"""
XSS tracer
The baseline-mlp model implementation.

Written by Jiaxi Wu
"""
import datetime
import os
import re
import keras.layers as KL


###############################################
#  Utility Functions
###############################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    print it's shapre, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shapre: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".
                     format(array.min(), array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("", ""))
        text += "  {}".format(array.dtype)
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None:  Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True:  (don't' use). Set layer in training mode even when making
                   inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


###############################################
#  Data Genereator
###############################################

def load_data():
    pass


###############################################
#  BaselineMLP Class
###############################################

class BaselineMLP():
    """Encapsulates the BaselineMLP model functionality.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build BaselineMLP architecture.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A ample model path might look list:
            # \path\to\logs\baseline20181219T2250\baseline_001.h5 (Windows)
            # /path/to/logs/baseline20181219T2250/baseline_001.ht5 (Linux)
            regex = r""".*[/\\][w-]+
                    (\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]
                    baseline_[\w-]+(\d{4})\.h5
                    """
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(
                    int(m.group(1)), int(m.group(2)), int(m.group(3)),
                    int(m.group(4)), int(m.group(5)),
                )
                # Epoch number in file is 1-based,
                #   and in Keras code it's 0-based.
                # So, adjust for that then increment
                #   by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch {}'.format(self.epoch))

        # Directory for training logs
        self.log_dir = os.path.join(
            self.model_dir, "{}{:%Y%m%dT%H%M}".
            format(self.config.NAME.lower(), now),
        )

        # Path to save after each epoch.
        #   Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(
            self.log_dir, "baseline_{}_*epoch*.h5".format(
                self.config.NAME.lower(),
            ),
        )
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}",
        )
