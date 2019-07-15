import os
from Base import Model, Utility
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import numpy as np
import matplotlib.pyplot as plt


strProjectFolder = os.path.dirname(os.path.dirname(__file__))
strOutputFolder = os.path.join(strProjectFolder, "03-Output")

history = Utility.recordsAccLossHistory()
def getTrain(dictModelPara, dictHyperPara, arrayTrain, arrayValid):

    model = Model.RNN(dictModelPara)

    callbacks = [EarlyStopping("val_loss", patience=20)
               , ModelCheckpoint(os.path.join(strOutputFolder, "model.h5"), save_best_only=True)
               , CSVLogger(os.path.join(strOutputFolder, "log.csv"), separator=",", append=False)
               , history]

    model.fit(x=arrayTrain[0], y=arrayTrain[1], epochs=dictHyperPara["intEpochs"], batch_size=dictHyperPara["intBatchSize"], verbose=2, validation_data=arrayValid, callbacks=callbacks)

    history.plotLosss()
    model.save(os.path.join(strOutputFolder, "model.h5"))

