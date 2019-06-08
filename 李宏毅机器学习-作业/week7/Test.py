import os
import numpy as np
import pandas as pd
from Base import DataProcessing, Train, Predict


def getTest():

    strProjectFolder = os.path.dirname(__file__)
    strRAWDataFolder = os.path.join(strProjectFolder, "01-RAWData")
    strAPDataFolder = os.path.join(strProjectFolder, "02-APData")
    strOutputFolder = os.path.join(strProjectFolder, "03-Output")
    submisson = pd.read_csv(os.path.join(strRAWDataFolder, "sampleSubmission.csv"))

    # Hyper Parameter
    intEpochs = 10
    intBatchSize = 256
    dictHyperPara = {"intEpochs":intEpochs, "intBatchSize":intBatchSize}

    # Model Parameter
    cell = "LSTM"
    intSequenceLength = 40
    intVocabSize = 10000
    intEmbeddingDim = 256
    intHiddenSize = 128
    floatDropoutRate = 0.5
    dictModelPara = {"cell":cell, "intSequenceLength":intSequenceLength, "intVocabSize":intVocabSize, "intEmbeddingDim":intEmbeddingDim, "intHiddenSize":intHiddenSize, "floatDropoutRate":floatDropoutRate}

    ETL = DataProcessing.executeETL()
    ETL.cleanData(strDataFileName="training_label.txt", boolLabel=True)
    ETL.loadTokenizer(strTokenizerFileName="TokenizerDictionary")
    ETL.convertWords2Sequence(intSequenceLength=intSequenceLength)
    arrayTrain = ETL.getData()

    Train.getTrain(dictModelPara=dictModelPara, dictHyperPara=dictHyperPara, arrayTrain=arrayTrain, arrayValid=arrayTrain)
    
    ETL.cleanData(strDataFileName="testing_data.txt", boolLabel=False)
    ETL.convertWords2Sequence(intSequenceLength=intSequenceLength)
    arrayTest = ETL.getData()

    arrayPredict = Predict.makePredict(arrayTest, strOutputFolder)

    submisson["label"] = pd.DataFrame(arrayPredict)
    submisson.to_csv(os.path.join(strOutputFolder, "submission.csv"), index=False)


if __name__ == "__main__":
    getTest()