import os
import pandas as pd
import numpy as np
from Utility import countLabelCorrespondMovieNum, countSingleLabelCorrespondMovieNum


strProjectFolder = os.path.dirname(os.path.dirname(__file__))

DataUser = pd.read_csv(os.path.join(strProjectFolder, "01-Data/users.csv"), sep="::", engine="python", usecols=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
DataMovie = pd.read_csv(os.path.join(strProjectFolder, "01-Data/movies.csv"), sep="::", engine="python", usecols=["movieID", "Title", "Genres"])
DataMovie = DataMovie.rename(index=str, columns={'movieID':'MovieID'})

# define new label
# Format, Setting, Mood
listClasses = ["Animation|Children's|Drama|Musical|Documentary", "Action|Adventure|Fantasy|Sci-Fi|Thriller|War|Mystery", "Comedy|Romance|Crime|Horror|Film-Noir"]

dictGenres = countLabelCorrespondMovieNum(DataMovie["Genres"]) # 原始分類電影數
dictSingleGenres =  countSingleLabelCorrespondMovieNum(DataMovie["Genres"]) # 個別一類電影數

# defin movie label
listMovieLabel = [] # 新的電影分類
for d in DataMovie["Genres"]:

    dictClassCount = {}
    for k in listClasses:dictClassCount[str(k)] = 0

    for c in listClasses:
        for g in d.split("|"):
            if g in c:
                dictClassCount[str(c)] += 1 
    strLabel = max(dictClassCount.items(), key = lambda x: x[1])[0]
    listMovieLabel.append(strLabel)

DataMovie["Label"] = listMovieLabel  
DataMovie.to_csv(os.path.join(strProjectFolder, "02-Output/movies.csv"))
DataUser.to_csv(os.path.join(strProjectFolder, "02-Output/users.csv"))
