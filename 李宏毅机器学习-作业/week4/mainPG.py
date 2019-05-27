import csv, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def getShuffleData(arrayX, arrayY):
    arrayRandomIndex = np.arange(len(arrayX))
    np.random.shuffle(arrayRandomIndex)
    return arrayX[arrayRandomIndex], arrayY[arrayRandomIndex]


def getTrainAndValidData(arrayTrainAllX, arrayTrainAllY, percentage):
    intInputDataSize = len(arrayTrainAllX)
    intValidDataSize = int(np.floor(intInputDataSize * percentage))

    arrayTrainAllX, arrayTrainAllY = getShuffleData(arrayTrainAllX, arrayTrainAllY)

    arrayValidX = arrayTrainAllX[0:intValidDataSize]
    arrayTrainX = arrayTrainAllX[intValidDataSize:]

    arrayValidY = arrayTrainAllY[0:intValidDataSize]
    arrayTrainY = arrayTrainAllY[intValidDataSize:]
    return arrayTrainX, arrayTrainY, arrayValidX, arrayValidY


def getSigmoidValue(z):
    s = 1 / (1.0 + np.exp(-z))
    return np.clip(s, 1e-8, 1 - (1e-8))


if __name__ == "__main__":

    np.random.seed(11)
    
    # read Training data, Training label, Testing data
    dfTrainX = pd.read_csv(os.path.join(os.path.dirname(__file__), "01-Data/X_train_my.csv"))
    dfTrainY = pd.read_csv(os.path.join(os.path.dirname(__file__), "01-Data/Y_train_my.csv"))
    dfTestX = pd.read_csv(os.path.join(os.path.dirname(__file__), "01-Data/X_test_my.csv"))

    # transform the data to array
    arrayTrainX = np.array(dfTrainX.values) # (32561, 106)
    arrayTestX = np.array(dfTestX.values) # (16281, 106)
    arrayTrainY = np.array(dfTrainY.values) # (32561)

    # take some training data to be validation data
    arrayTrainX, arrayTrainY, arrayValidX, arrayValidY = getTrainAndValidData(arrayTrainAllX=arrayTrainX, arrayTrainAllY=arrayTrainY, percentage=0.3)

    # calculate maximum likelihood esitimator of mu and sigma
    intTrainSize = arrayTrainX.shape[0]
    intCount1 = 0
    intCount2 = 0

    arrayMu1 = np.zeros(arrayTrainX.shape[1]) # (106, )
    arrayMu2 = np.zeros(arrayTrainX.shape[1]) # (106, )
    for idx in range(intTrainSize):
        if arrayTrainY[idx] == 1:
            arrayMu1 += arrayTrainX[idx]
            intCount1 += 1
        else:    
            arrayMu2 += arrayTrainX[idx]
            intCount2 += 1

    arrayMu1 /= intCount1  
    arrayMu2 /= intCount2   

    arraySigma1 = np.zeros((arrayTrainX.shape[1], arrayTrainX.shape[1])) # (106, 106)
    arraySigma2 = np.zeros((arrayTrainX.shape[1], arrayTrainX.shape[1])) # (106, 106)
    for idx in range(intTrainSize):
        if arrayTrainY[idx] == 1:
            arraySigma1 += np.dot(np.transpose([arrayTrainX[idx]-arrayMu1]), [arrayTrainX[idx]-arrayMu1])
            # arraySigma1 += np.dot(np.transpose((arrayTrainX[idx]-arrayMu1)), (arrayTrainX[idx]-arrayMu1)) #can not inv
        else:    
            arraySigma2 += np.dot(np.transpose([arrayTrainX[idx]-arrayMu2]), [arrayTrainX[idx]-arrayMu2])
            # arraySigma2 += np.dot(np.transpose((arrayTrainX[idx]-arrayMu2)), (arrayTrainX[idx]-arrayMu2)) #can not inv
    
    arrayCovariance = (float(intCount1)/intTrainSize) * arraySigma1 + (float(intCount2)/intTrainSize) * arraySigma2

    arrayCovarianceInverse = np.linalg.inv(arrayCovariance)


    # validation
    arrayW = np.dot(np.transpose((arrayMu1 - arrayMu2)), arrayCovarianceInverse)
    arrayB = -(0.5) * np.dot(np.dot(np.transpose(arrayMu1), arrayCovarianceInverse), arrayMu1) + (0.5) * np.dot(np.dot(np.transpose(arrayMu2), arrayCovarianceInverse), arrayMu2) + np.log(float(intCount1/intCount2))

    z = np.dot(arrayW, np.transpose(arrayValidX)) + arrayB
    s = getSigmoidValue(z)
    result = ((np.around(s)) == np.squeeze(arrayValidY))
    print("Vaild Accuracy:{} ".format(float(result.sum())/ len(arrayValidY)))


    # test
    ans = pd.read_csv(os.path.join(os.path.dirname(__file__), "correct_answer.csv"))
    z = np.dot(arrayW, np.transpose(arrayTestX)) + arrayB
    predict = np.around(getSigmoidValue(z))
    result = (predict == np.squeeze(ans["label"]))
    print("Test Accuracy:{} ".format(float(result.sum())/ len(arrayTestX)))


