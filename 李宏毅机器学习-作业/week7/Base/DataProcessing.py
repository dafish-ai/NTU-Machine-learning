import os, json
from keras.preprocessing.text import Tokenizer
import _pickle as pk
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical


strProjectFolder = os.path.dirname(os.path.dirname(__file__))
strRAWDataFolder = os.path.join(strProjectFolder, "01-RAWData")
strAPDataFolder = os.path.join(strProjectFolder, "02-APData")

class executeETL():
    def __init__(self):
        self.dictData = {}

    def cleanData(self, strDataFileName, boolLabel):
        listLabel = []
        listText = []
        with open(os.path.join(strRAWDataFolder, strDataFileName), "r", encoding="utf8") as data:
            for d in data:
                if boolLabel:
                    listRow = d.strip().split(" +++$+++ ")
                    listLabel.append(int(listRow[0]))
                    listText.append(listRow[1])
                else:
                    listRow = d.strip().split(",", 1)[1]
                    if listRow != "text":
                        listText.append(listRow)

            if boolLabel:
                self.dictData["Data"] = [listText, listLabel]
            else:
                self.dictData["Data"] = [listText]

    def doTokenizer(self, intVocabSize):
        self.tokenizer = Tokenizer(num_words=intVocabSize)
        for key in self.dictData:
            listTexts = self.dictData[key][0]
            self.tokenizer.fit_on_texts(listTexts)

    def saveTokenizer(self, strTokenizerFileName):
        pk.dump(self.tokenizer, open(os.path.join(strAPDataFolder, strTokenizerFileName), "wb"))

    def loadTokenizer(self, strTokenizerFileName):
        self.tokenizer = pk.load(open(os.path.join(strAPDataFolder, strTokenizerFileName), "rb"))

    def convertWords2Sequence(self, intSequenceLength):
        for key in self.dictData:
            listSequence = self.tokenizer.texts_to_sequences(self.dictData[key][0])
            print("text count start")
            listTextCount = []
            for t in listSequence:
                listTextCount.append(len(t))     
            
            import pandas as pd
            print(pd.Series(listTextCount).value_counts())         

            self.dictData[key][0] = np.array(pad_sequences(listSequence, maxlen=intSequenceLength))
    
    def convertLabel2Onehot(self):
        for key in self.dictData:
            if len(self.dictData[key]) == 2:
                self.dictData[key][1] = np.array(to_categorical(self.dictData[key][1]))

    def splitData(self, floatRatio):
        data = self.dictData["Data"]
        X = data[0]
        Y = data[1]
        intDataSize = len(X)
        intValidationSize = int(intDataSize * floatRatio)
        return (X[intValidationSize:], Y[intValidationSize:]), (X[:intValidationSize], Y[:intValidationSize])

    def getData(self):
        return self.dictData["Data"]

