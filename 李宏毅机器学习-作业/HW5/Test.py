import os
import numpy as np
import pandas as pd
from Base import Train, Predict


def getTest(boolNormalize, boolDeep, boolBias, strProjectFolder):

    if boolNormalize:
        if boolDeep:
            strOutputPath = "02-Output/" + "Deep" + "Normal"
        else:
            if boolBias:
                strOutputPath = "02-Output/" + "Bias" + "Normal"
            else:
                strOutputPath = "02-Output/" + "unBias" + "Normal"
    else:
        if boolDeep:
            strOutputPath = "02-Output/" + "Deep" 
        else:
            if boolBias:
                strOutputPath = "02-Output/" + "Bias" 
            else:
                strOutputPath = "02-Output/" + "unBias"
    
    strOutputPath = strOutputPath + "Test"

    DataTrain = pd.read_csv(os.path.join(strProjectFolder, "01-Data/Train.csv"))
    DataTest = pd.read_csv(os.path.join(strProjectFolder, "01-Data/Test.csv"))
    submisson = pd.read_csv(os.path.join(strProjectFolder, "01-Data/SampleSubmisson.csv"))

    DataTrain = DataTrain.sample(frac=1)
    intUserSize = len(DataTrain["UserID"].drop_duplicates())
    intMovieSize = len(DataTrain["MovieID"].drop_duplicates())

    arrayUsers = DataTrain["UserID"].values
    arrayMovies = DataTrain["MovieID"].values
    arrayRate = DataTrain["Rating"].values

    arrayTestUsers = DataTest["UserID"].values
    arrayTestMovies = DataTest["MovieID"].values

    intLatentSize = 32

    if boolNormalize:
        arrayRateAvg = np.mean(arrayRate)
        arrayRateStd = np.std(arrayRate)
        arrayRate = (arrayRate - arrayRateAvg)/arrayRateStd

    Train.getTrain(arrayTrainUser=arrayUsers, arrayTrainMovie=arrayMovies, arrayTrainRate=arrayRate
                 , arrayValidUser=arrayUsers, arrayValidMovie=arrayMovies, arrayValidRate=arrayRate
                 , intUserSize=intUserSize
                 , intMovieSize=intMovieSize
                 , intLatentSize=intLatentSize
                 , boolBias=boolBias
                 , boolDeep=boolDeep
                 , strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)

    arrayPredict = Predict.makePredict(arrayTestUsers, arrayTestMovies, strProjectFolder, strOutputPath)

    if boolNormalize:
        arrayPredict = (arrayPredict * arrayRateStd) + arrayRateAvg


    submisson["Rating"] = pd.DataFrame(arrayPredict)
    submisson.to_csv(os.path.join(strProjectFolder, strOutputPath + "submission.csv"), index=False)


if __name__ == "__main__":

    strProjectFolder = os.path.dirname(__file__)

    getTest(boolNormalize=True, boolDeep=False, boolBias=True, strProjectFolder=strProjectFolder)