import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import plot_model
from Base import Utility
from sklearn.manifold import TSNE


def plotModel(strProjectFolder, strOutputPath):
    """
    This function plots the model structure.
    """
    model = load_model(os.path.join(strProjectFolder, strOutputPath + "model.h5"), custom_objects={"getRMSE": Utility.getRMSE})
    model.summary()
    plot_model(model, show_shapes=True, to_file=os.path.join(strProjectFolder, strOutputPath + "model.png"))


def plotLossAccuracyCurves(strProjectFolder, strOutputPath):
    """
    This function plots the Loss Curves
    """
    pdLog = pd.read_csv(os.path.join(strProjectFolder, strOutputPath + "log.csv"))

    fig = plt.figure(figsize=(6, 4))
    # Loss Curves
    plt.plot(pdLog["epoch"], pdLog["getRMSE"], "r", linewidth=1.5)
    plt.plot(pdLog["epoch"], pdLog["val_getRMSE"], "b", linewidth=1.5)
    plt.legend(["Training RMSE", "Validation RMSE"], fontsize=12)
    plt.xlabel("Epochs ", fontsize=10)
    plt.ylabel("RMSE Loss", fontsize=10)
    plt.title("Loss Curves", fontsize=10)
    plt.savefig(os.path.join(strProjectFolder, strOutputPath + "LossCurves"))


def plotMovieEmbeddingTSNE(dictLabel, strProjectFolder, strOutputPath):
    """
    This function Visualizate the MovieEmbedding with label arrayY
    """
    model = load_model(os.path.join(strProjectFolder, strOutputPath + "model.h5"), custom_objects={"getRMSE": Utility.getRMSE})
    arrayMovieEmbedding = np.array(model.layers[3].get_weights()).squeeze()

    arrayVisualizationData = TSNE(n_components=2).fit_transform(arrayMovieEmbedding)
    arrayVisualizationX = arrayVisualizationData[:, 0]
    arrayVisualizationY = arrayVisualizationData[:, 1]

    fig = plt.figure(figsize=(6, 6))
    cm = plt.cm.get_cmap("RdYlBu")
    plt.scatter(arrayVisualizationX, arrayVisualizationY, c=dictLabel["LabelEncoder"], cmap=cm)
    plt.colorbar()
    plt.savefig(os.path.join(strProjectFolder, strOutputPath + "TSNE"))
