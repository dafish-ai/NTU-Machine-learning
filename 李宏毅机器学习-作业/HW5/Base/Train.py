import os
import numpy as np
from Base import Model, Plot
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


def getTrain(arrayTrainUser, arrayTrainMovie, arrayTrainRate, arrayValidUser, arrayValidMovie, arrayValidRate, intUserSize, intMovieSize, intLatentSize, boolBias, boolDeep, strProjectFolder, strOutputPath):

    if boolDeep:
        model = Model.MFDeepNN(intUserSize=intUserSize, intMovieSize=intMovieSize, intLatentSize=intLatentSize)
    else:   
        model = Model.MF(intUserSize=intUserSize, intMovieSize=intMovieSize, intLatentSize=intLatentSize, boolBias=boolBias)

    callbacks = [EarlyStopping("val_loss", patience=100)
               , ModelCheckpoint(os.path.join(strProjectFolder, strOutputPath + "model.h5"), save_best_only=True)
               , CSVLogger(os.path.join(strProjectFolder, strOutputPath + "log.csv"), separator=",", append=False)]
    
    model.fit([arrayTrainUser, arrayTrainMovie], arrayTrainRate, epochs=100, batch_size=4096, verbose=2, validation_data=([arrayValidUser, arrayValidMovie], arrayValidRate), callbacks=callbacks)

    model.save(os.path.join(strProjectFolder, strOutputPath + "model.h5"))

    