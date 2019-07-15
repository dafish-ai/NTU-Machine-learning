import csv, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def getShuffleData(arrayX, arrayY):
    arrayRandomIndex = np.arange(len(arrayX))
    np.random.shuffle(arrayRandomIndex)
    return arrayX[arrayRandomIndex], arrayY[arrayRandomIndex]


def getNormalizeData(arrayTrainX, arrayTestX):
    arrayX = np.concatenate((arrayTrainX, arrayTestX))
    
    arrayMuX = np.mean(arrayX, axis=0)
    arraySigmaX = np.std(arrayX, axis=0)

    arrayNormalizeX = (arrayX - arrayMuX) / arraySigmaX

    arrayNormalizeTrainX, arrayNormalizeTestX = arrayNormalizeX[0:arrayTrainX.shape[0]], arrayNormalizeX[arrayTrainX.shape[0]:]
    return arrayNormalizeTrainX, arrayNormalizeTestX


def getTrainAndValidData(arrayTrainAllX, arrayTrainAllY, percentage):
    intInputDataSize = len(arrayTrainAllX)
    intValidDataSize = int(np.floor(intInputDataSize * percentage))

    arrayTrainAllX, arrayTrainAllY = getShuffleData(arrayTrainAllX, arrayTrainAllY)

    arrayValidX = arrayTrainAllX[0:intValidDataSize]
    arrayTrainX = arrayTrainAllX[intValidDataSize:]

    arrayValidY = arrayTrainAllY[0:intValidDataSize]
    arrayTrainY = arrayTrainAllY[intValidDataSize:]
    return arrayTrainX, arrayTrainY, arrayValidX, arrayValidY


def getLinear(arrayX, arrayW, arrayB):
    X = np.dot(arrayX, arrayW) + arrayB
    return X


def getSigmoidValue(z):
    s = 1 / (1.0 + np.exp(-z))
    return np.clip(s, 1e-8, 1 - (1e-8))


def getCrossEntropyValue(s, Y):
    arrayCrossEntropy = -1 * (np.dot(np.transpose(Y), np.log(s)) + np.dot((1-np.transpose(Y)), np.log(1-s))) / len(Y)
    return arrayCrossEntropy


def makePredict(s):
    pre = np.around(s)
    return pre


def computeAccuracy(Predict, Y):
    result = (Predict == np.squeeze(Y))
    Accuracy = sum(result) / len(Y)
    return Accuracy


def trainMBGD(arrayTrainX, arrayTrainY, intBatchSize, intEpochSize, floatLearnRate):
    
    arrayW = np.zeros((arrayTrainX.shape[1])) # (106, )
    arrayB = np.zeros(1) # (1, )

    arrayTotalLoss = 0.0
    listTrainCost = []
    listValidAccuracy = []
    listValidCost = []
    for epoch in range(intEpochSize):

        if epoch > 0:
            # train cost
            arrayTrainCost = arrayTotalLoss / (len(arrayTrainX))
            listTrainCost.append(arrayTrainCost)
            print("Epoch:{}, Epoch average loss:{} ".format(epoch, arrayTrainCost))
            
            # vaild cost
            z = getLinear(arrayValidX, arrayW, arrayB)
            # z = np.dot(arrayValidX, arrayW) + arrayB
            s = getSigmoidValue(z)
            arrayPredict = makePredict(s)
            arrayVaildCrossEntropy = getCrossEntropyValue(s, arrayValidY)
            listValidCost.append(arrayVaildCrossEntropy)

            # vaild accuracy
            arrayValidAccuracy = computeAccuracy(arrayPredict, arrayValidY)
            listValidAccuracy.append(arrayValidAccuracy)
            print("Validition Accuracy:{} ".format(arrayValidAccuracy))
            
            arrayTotalLoss = 0.0

        arrayTrainX, arrayTrainY = getShuffleData(arrayX=arrayTrainX, arrayY=arrayTrainY)

        for batch_iter in range(int(len(arrayTrainX)/intBatchSize)):
            X = arrayTrainX[intBatchSize*batch_iter:intBatchSize*(batch_iter+1)] # (intBatchSize, 106)
            Y = arrayTrainY[intBatchSize*batch_iter:intBatchSize*(batch_iter+1)] # (intBatchSize, 1)

            z = getLinear(X, arrayW, arrayB)
            # z = np.dot(X, arrayW) + arrayB
            s = getSigmoidValue(z)

            arrayCrossEntropy = getCrossEntropyValue(s, Y) * len(Y)
            arrayTotalLoss += arrayCrossEntropy

            # arrayGradientW = np.mean(-1 * X * (np.squeeze(Y) - s).reshape((intBatchSize,1)), axis=0) # need check
            arrayGradientW = -1 * np.dot(np.transpose(X), (np.squeeze(Y) - s).reshape((intBatchSize,1))) 
            arrayGradientB = np.mean(-1 * (np.squeeze(Y) - s))
        
            arrayW -= floatLearnRate * np.squeeze(arrayGradientW)
            arrayB -= floatLearnRate * arrayGradientB

        # print("CrossEntropy:{} , TotalLoss{} ".format(arrayCrossEntropy, arrayTotalLoss))

    plt.plot(np.arange(len(listValidCost)), listValidCost, "r--", label="Vaild Cost")
    plt.plot(np.arange(len(listTrainCost)), listTrainCost, "b--", label="Train Cost")
    plt.title("Train Process")
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function (Cross Entropy)")
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), "02-Output/TrainProcess"))
    plt.show()
    return arrayW, arrayB


if __name__ == "__main__":

    # np.random.seed(1)

    # read Training data, Training label, Testing data
    dfTrainX = pd.read_csv(os.path.join(os.path.dirname(__file__), "01-Data/X_train_my.csv"))
    dfTrainY = pd.read_csv(os.path.join(os.path.dirname(__file__), "01-Data/Y_train_my.csv"))
    dfTestX = pd.read_csv(os.path.join(os.path.dirname(__file__), "01-Data/X_test_my.csv"))

    # transform the data to array
    arrayTrainX = np.array(dfTrainX.values) # (32561, 106)
    arrayTestX = np.array(dfTestX.values) # (16281, 106)
    arrayTrainY = np.array(dfTrainY.values) # (32561)

    # normalize the Training and Testing data
    arrayNormalizeTrainX, arrayNormalizeTestX = getNormalizeData(arrayTrainX, arrayTestX)

    # shuffling data index
    arrayTrainAllX, arrayTrainAllY = getShuffleData(arrayNormalizeTrainX, arrayTrainY)

    # take some training data to be validation data
    arrayTrainX, arrayTrainY, arrayValidX, arrayValidY = getTrainAndValidData(arrayTrainAllX, arrayTrainAllY, 0.2)

    ###---Train(mini batch gradient descent)---###
    arrayW, arrayB = trainMBGD(arrayTrainX, arrayTrainY, intBatchSize=32, intEpochSize=300, floatLearnRate=0.001)

    ###---Test---###
    ans = pd.read_csv(os.path.join(os.path.dirname(__file__), "02-Output/correct_answer.csv"))
    arrayTestZ = getLinear(arrayNormalizeTestX, arrayW, arrayB)
    # Testz = (np.dot(arrayNormalizeTestX, arrayW) + arrayB)
    arrayPredict = makePredict(getSigmoidValue(arrayTestZ))

    dictD = {"Predict":arrayPredict, "Target":ans["label"]}
    pdResult = pd.DataFrame(dictD, columns=dictD.keys())
    pdResult.to_csv(os.path.join(os.path.dirname(__file__), "02-Output/Predict"))
    # print(pdResult)

    result = computeAccuracy(arrayPredict, np.squeeze(ans["label"]))
    print("Testing Accuracy:{} ".format(result))

