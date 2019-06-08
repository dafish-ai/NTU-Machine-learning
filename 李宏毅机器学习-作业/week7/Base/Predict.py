import os
import numpy as np
from keras.models import load_model


def makePredict(arrayTest, strOutputFolder):

    strModelPath = os.path.join(strOutputFolder, "model.h5")
    
    model = load_model(strModelPath)

    predictions = model.predict(arrayTest[0], batch_size=256)
    predictions = (predictions > 0.5).astype(np.int)

    return predictions