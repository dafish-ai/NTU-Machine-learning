import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import time


strProjectPath = os.path.dirname(__file__)
strRAWDataPath = os.path.join(strProjectPath, "01-RAWData")

strFaceDataPath = os.path.join(strRAWDataPath, "Aberdeen")
strFaceData = [os.path.join(strFaceDataPath, img) for img in os.listdir(strFaceDataPath)]
strOutputPath = os.path.join(strProjectPath, "Output/pca")

class prepareImage():
    def __init__(self, int_Image_Size):
        self.list_Images_Vector = []
        self.int_Image_Size = int_Image_Size

    def resizeImage(self):
        """
        This function can resize the image. 
        In this process, it will do normalize for each image then resize the image.
        """
        for image in strFaceData:
            arrayImage = io.imread(image)
            arrayResizeImage = transform.resize(arrayImage, (self.int_Image_Size, self.int_Image_Size, 3))
            self.list_Images_Vector.append(arrayResizeImage.flatten())


class PCA(object):
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, X):
        X = self._check_array(X) # (N, D)

        self.mean_ = self._check_array(np.mean(X, axis=0)) # (1, D)

        U, S, V = np.linalg.svd((X - self.mean_).T, full_matrices=False)

        self.S = S[0:self.n_components]
        self.U = U[:, 0:self.n_components] # (D, n_components)
        self.explained_variance_ratio_ = np.round(self.S / np.sum(self.S), 3)
        print(self.explained_variance_ratio_)
        return
    
    def transform(self, X):
        X = self._check_array(X)
        X -= self.mean_

        Z = np.dot(X, self.U)
        return Z
    
    def inverse_transform(self, X):
        Z = self.transform(X) # (n, n_componet)

        X_hat = np.dot(Z, self.U.T) #(n, D)
        return X_hat

    def _check_array(self, ndarray):
        ndarray = np.array(ndarray)
        if len(ndarray.shape) == 1: 
            ndarray = np.reshape(ndarray, (1, ndarray.shape[0]))
        return ndarray


def get_Img_Clip(img):
    img -= np.min(img)
    img /= np.max(img)
    img = (img * 255).astype(np.uint8)
    img = np.reshape(img, (128, 128, 3))
    return img


if __name__ == "__main__":
    processing = prepareImage(int_Image_Size=128)
    processing.resizeImage()
    list_Images_Vector = processing.list_Images_Vector # NxD

    pca = PCA(n_components=4)
    pca.fit(X=list_Images_Vector)

    list_Random_Index = [100, 200, 128, 400]
    list_Random_Images_Vectors = [list_Images_Vector[i] for i in list_Random_Index]
    array_Recon_Images_Vectors = pca.inverse_transform(X=list_Random_Images_Vectors)
    ReconImage = array_Recon_Images_Vectors + pca.mean_

    # plot avg face
    io.imsave(os.path.join(strOutputPath, "AvgFace.png"), pca.mean_.reshape(128, 128, 3))

    # plot top 4 eigen face
    for i in range(4):
        io.imsave(os.path.join(strOutputPath, "Top{}EigenFaces.png".format(i)), get_Img_Clip(pca.U[:, i]))
    
    # plot 4 random reconstruct images
    for i in range(len(list_Random_Index)):
        io.imsave(os.path.join(strOutputPath, "Recon_img{}_{}Eigen.png".format(list_Random_Index[i], pca.n_components)), get_Img_Clip(ReconImage[i]))
        io.imsave(os.path.join(strOutputPath, "Origin_img{}.png".format(list_Random_Index[i])), list_Images_Vector[list_Random_Index[i]].reshape(128, 128, 3))

