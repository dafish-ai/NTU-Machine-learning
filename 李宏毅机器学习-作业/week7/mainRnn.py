import os
from Base import DataProcessing, Train


strProjectFolder = os.path.dirname(__file__)
strAPDataFolder = os.path.join(strProjectFolder, "02-APData")
strOutputFolder = os.path.join(strProjectFolder, "03-Output")

floatRatio = 0.1

# Hyper Parameter
intEpochs = 10
intBatchSize = 512
dictHyperPara = {"intEpochs":intEpochs, "intBatchSize":intBatchSize}

# Model Parameter
intSequenceLength = 40
intVocabSize = 20000
intEmbeddingDim = 128
intHiddenSize = 512
floatDropoutRate = 0.3
dictModelPara = {"intSequenceLength":intSequenceLength, "intVocabSize":intVocabSize, "intEmbeddingDim":intEmbeddingDim, "intHiddenSize":intHiddenSize, "floatDropoutRate":floatDropoutRate}



ETL = DataProcessing.executeETL()
ETL.cleanData(strDataFileName="training_label.txt", boolLabel=True)

if "TokenizerDictionary" not in os.listdir(strAPDataFolder):
    ETL.doTokenizer(intVocabSize=intVocabSize)
    ETL.saveTokenizer(strTokenizerFile="TokenizerDictionary")
else:
    ETL.loadTokenizer(strTokenizerFile="TokenizerDictionary")

ETL.convertWords2Sequence(intSequenceLength=intSequenceLength)

arrayTrain, arrayValid = ETL.splitData(floatRatio=floatRatio)

Train.getTrain(dictModelPara=dictModelPara, dictHyperPara=dictHyperPara, arrayTrain=arrayTrain, arrayValid=arrayValid)





