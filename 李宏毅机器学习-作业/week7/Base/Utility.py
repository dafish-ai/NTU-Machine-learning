import os
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt


strProjectFolder = os.path.dirname(os.path.dirname(__file__))
strOutputFolder = os.path.join(strProjectFolder, "03-Output")

class recordsAccLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.val_losses = []
        self.val_accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs["loss"])
        self.accuracy.append(logs["acc"])
        self.val_losses.append(logs["val_loss"])
        self.val_accuracy.append(logs["val_acc"])

    def plotLosss(self):
        fig = plt.figure(figsize=(12, 5))
        # Loss Curves
        ax = fig.add_subplot(1, 2, 1)
        plt.plot(np.arange(len(self.losses)), self.losses, label="losses")
        plt.plot(np.arange(len(self.val_losses)), self.val_losses, label="val_losses")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.title("loss process")
        plt.tight_layout()
        # Accuracy Curves
        ax = fig.add_subplot(1, 2, 2)
        plt.plot(np.arange(len(self.accuracy)), self.accuracy, label="accuracy")
        plt.plot(np.arange(len(self.val_accuracy)), self.val_accuracy, label="val_accuracy")
        plt.xlabel("epochs")
        plt.ylabel("acc")
        plt.legend()
        plt.title("accuracy process")
        plt.tight_layout()
        plt.savefig(os.path.join(strOutputFolder, "LossAccuracyCurves"))