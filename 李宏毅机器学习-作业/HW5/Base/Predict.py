import os
from keras.models import load_model
from Base import Utility


def makePredict(arrayUser, arrayMovie, strProjectFolder, strOutputPath):

    strModelPath = os.path.join(strProjectFolder, strOutputPath + "model.h5")
    
    model = load_model(strModelPath, custom_objects={"getRMSE": Utility.getRMSE})

    predictions = model.predict([arrayUser, arrayMovie], batch_size=1024)

    return predictions
