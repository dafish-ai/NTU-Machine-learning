import keras.backend as K
import numpy as np


def getRMSE(arrayPredict, arrayTrue):
    return K.sqrt(K.mean(K.pow(arrayTrue - arrayPredict, 2))) 


def getLabelEncoder(DataMovies):
    '''
    This function finds the Movie correspond Label and return a dictionary order by MovieID from trainind data
    '''          
    from sklearn.preprocessing import LabelEncoder
    LE = LabelEncoder()
    DataMovies["LabelEncoder"] = LE.fit_transform(DataMovies["Label"])
    return DataMovies


def countLabelCorrespondMovieNum(pdDataLableColumn):
    '''
    This function counts the number of movies in each label
    '''
    dictGenres = {} 
    for d in pdDataLableColumn:
        if d not in dictGenres.keys():
            dictGenres[str(d)] = 1
        else:
            dictGenres[str(d)] += 1
    return dictGenres


def countSingleLabelCorrespondMovieNum(pdDataLableColumn):
    '''
    This function counts the number of movies in each Single label
    '''
    dictSingleGenres = {} # 個別一類電影數
    listSingleGenres = [] # 電影個別一類分類
    for d in pdDataLableColumn:
        for g in d.split("|"):
            if g not in listSingleGenres:
                listSingleGenres.append(g)
                dictSingleGenres[str(g)] = 1
            else:
                dictSingleGenres[str(g)] += 1
    return dictSingleGenres