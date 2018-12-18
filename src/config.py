# coding: utf-8
"""
XSS tracer
Base Configurations class.

Written by Jiaxi Wu
"""
import os


# Base Configuration Classes
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """ Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'baseline', 'Experiment 1', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    # get working dir and define for self
    WORKING_DIR = os.path.split(os.path.realpath(__file__))[0]

    # define data sotre path
    DATA_DIR = os.path.join(WORKING_DIR, '../data/data_v1')
    # define data(x) file and labels(y) file
    DATA_X_FILE = os.path.join(DATA_DIR, 'train.cs`v')
    DATA_Y_FILE = os.path.join(DATA_DIR, 'labels.csv')

    # define result dir and history dir
    RESULT_DIR = os.path.join(WORKING_DIR, '../result')
    HIS_FILE = os.path.join(RESULT_DIR, 'history.csv')

    # training parameters configuration

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Learning rate and momentum
    LEARNING_RATE = 0.01
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    def __init__(self):
        """Set values of computed attributes."""
        self.BATCH_SIZE = 128

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startwith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
            print("\n")
