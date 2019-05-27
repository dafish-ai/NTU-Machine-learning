import csv, os
import numpy as np
import pandas as pd

def makeDataProcessing(dfData):
    dfDataX = dfData.drop(["education_num", "sex"], axis=1)

    listObjectColumnName = [col for col in dfDataX.columns if dfDataX[col].dtypes=="object"]
    listNonObjectColumnName = [col for col in dfDataX.columns if dfDataX[col].dtypes!="object"]

    dfNonObjectData = dfDataX[listNonObjectColumnName]
    dfNonObjectData.insert(2, "sex", (dfData["sex"]==" Male").astype(np.int)) # Male 1 Femal 0

    dfObjectData = dfDataX[listObjectColumnName]
    dfObjectData = pd.get_dummies(dfObjectData)

    dfDataX = dfNonObjectData.join(dfObjectData)
    dfDataX = dfDataX.astype("int64")
    return dfDataX

if __name__ == "__main__":

    # read raw data
    dfDataTrain = pd.read_csv(os.path.join(os.path.dirname(__file__), "train.csv"))
    dfDataTest = pd.read_csv(os.path.join(os.path.dirname(__file__), "test.csv"))

    # show Training Size and Testing Size
    intTrainSize = len(dfDataTrain)
    intTestSize = len(dfDataTest)

    # processing Training Label (Y)
    dfDataTrainY = dfDataTrain["income"]
    dfTrainY = pd.DataFrame((dfDataTrainY==" >50K").astype("int64"), columns=["income"]) # >50K 1, =<50K 0

    # processing Training and Testing data (X)
    dfDataTrain = dfDataTrain.drop(["income"], axis=1)
    dfAllData = pd.concat([dfDataTrain, dfDataTest], axis=0, ignore_index=True)
    dfAllData = makeDataProcessing(dfData=dfAllData)

    # sperate All data to Training and Testing
    dfTrainX = dfAllData[0:intTrainSize]
    dfTestX = dfAllData[intTrainSize:(intTrainSize + intTestSize)]

    # save Training data, Testing data and Training label
    dfTrainX.to_csv(os.path.join(os.path.dirname(__file__), "X_train_my.csv"), index=False)
    dfTestX.to_csv(os.path.join(os.path.dirname(__file__), "X_Test_my.csv"), index=False)
    dfTrainY.to_csv(os.path.join(os.path.dirname(__file__), "Y_train_my.csv"), index=False)
