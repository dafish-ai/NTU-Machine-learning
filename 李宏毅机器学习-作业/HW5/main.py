import os
import numpy as np
import pandas as pd
from Base import Train, Predict, Plot, Utility


def main(boolNormalize, boolDeep, boolBias):
    strProjectFolder = os.path.dirname(__file__)

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

    DataTrain = pd.read_csv(os.path.join(strProjectFolder, "01-Data/train.csv"), usecols=["UserID", "MovieID", "Rating"])
    DataUser = pd.read_csv(os.path.join(strProjectFolder, "02-Output/users.csv"), usecols=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
    DataMovie = pd.read_csv(os.path.join(strProjectFolder, "02-Output/movies.csv"), usecols=["MovieID", "Title", "Genres", "Label"])
    DataMovie = Utility.getLabelEncoder(DataMovie)

    DataTrain = DataTrain.sample(frac=1.0, random_state=10)
    intUserSize = len(DataUser["UserID"].drop_duplicates())
    intMovieSize = len(DataMovie["MovieID"].drop_duplicates())

    arrayUsers = DataTrain["UserID"].values
    arrayMovies = DataTrain["MovieID"].values
    arrayRate = DataTrain["Rating"].values

    intLatentSize = 32
    intVaildSize = 80000
    arrayTrainUser = arrayUsers[:-intVaildSize]
    arrayTrainMovie = arrayMovies[:-intVaildSize]
    arrayTrainRate = arrayRate[:-intVaildSize]

    arrayValidUser = arrayUsers[-intVaildSize:]
    arrayValidMovie = arrayMovies[-intVaildSize:]
    arrayValidRate = arrayRate[-intVaildSize:]

    arrayTrainRateAvg = np.mean(arrayTrainRate)
    arrayTrainRateStd = np.std(arrayTrainRate)

    if boolNormalize:
        arrayTrainRate = (arrayTrainRate - arrayTrainRateAvg)/arrayTrainRateStd
        arrayValidRate = (arrayValidRate - arrayTrainRateAvg)/arrayTrainRateStd

    Train.getTrain(arrayTrainUser=arrayTrainUser, arrayTrainMovie=arrayTrainMovie, arrayTrainRate=arrayTrainRate
                , arrayValidUser=arrayValidUser, arrayValidMovie=arrayValidMovie, arrayValidRate=arrayValidRate
                , intUserSize=intUserSize
                , intMovieSize=intMovieSize
                , intLatentSize=intLatentSize
                , boolBias=boolBias
                , boolDeep=boolDeep
                , strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)

    Plot.plotModel(strProjectFolder, strOutputPath)
    Plot.plotLossAccuracyCurves(strProjectFolder, strOutputPath)

    if not boolDeep:
        Plot.plotMovieEmbeddingTSNE(DataMovie, strProjectFolder, strOutputPath)

    arrayTrainPredict = Predict.makePredict(arrayTrainUser, arrayTrainMovie, strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)
    if boolNormalize:
        arrayTrainPredict = (arrayTrainPredict * arrayTrainRateStd) + arrayTrainRateAvg
    trainMSE = np.mean(np.square(np.squeeze(arrayTrainPredict) - arrayRate[:-intVaildSize]))
    trainRMSE = np.sqrt(np.mean(np.square(np.squeeze(arrayTrainPredict) - arrayRate[:-intVaildSize])))

    arrayValidPredict = Predict.makePredict(arrayValidUser, arrayValidMovie, strProjectFolder=strProjectFolder, strOutputPath=strOutputPath)
    if boolNormalize:
        arrayValidPredict = (arrayValidPredict * arrayTrainRateStd) + arrayTrainRateAvg
    ValidMSE = np.mean(np.square(np.squeeze(arrayValidPredict) - arrayRate[-intVaildSize:]))
    ValidRMSE = np.sqrt(np.mean(np.square(np.squeeze(arrayValidPredict) - arrayRate[-intVaildSize:])))
    
    print(trainMSE, trainRMSE, ValidMSE, ValidRMSE)

if __name__ == "__main__":
    main(boolNormalize=True, boolDeep=True, boolBias=False)



