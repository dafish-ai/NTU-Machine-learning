from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.layers import Input, Embedding, Flatten, Dot, Add, Dropout, Concatenate, Dense
from keras.optimizers import Adam
from Base import Utility


def MF(intUserSize, intMovieSize, intLatentSize, boolBias):
    UserInput = Input(shape=(1, ), name="UserInput")
    UserEmbedding = Embedding(intUserSize, intLatentSize, embeddings_initializer="random_normal", embeddings_regularizer=l2(0.001), name="UserEmbedding")(UserInput)
    UserEmbedding = Flatten()(UserEmbedding)
    UserEmbedding = Dropout(0.3)(UserEmbedding)

    MovieInput = Input(shape=(1, ), name="MovieInput")
    MovieEmbedding = Embedding(intMovieSize, intLatentSize, embeddings_initializer="random_normal", embeddings_regularizer=l2(0.001), name="MovieEmbedding")(MovieInput)
    MovieEmbedding = Flatten()(MovieEmbedding)
    MovieEmbedding = Dropout(0.3)(MovieEmbedding)

    RattingHat = Dot(axes=1)([UserEmbedding, MovieEmbedding])

    if boolBias:
        UserBias = Embedding(intUserSize, 1, embeddings_initializer="zeros", name="UserBias")(UserInput)
        UserBias = Flatten()(UserBias)

        MovieBias = Embedding(intMovieSize, 1, embeddings_initializer="zeros", name="MovieBias")(MovieInput)
        MovieBias = Flatten()(MovieBias)

        RattingHat = Add()([RattingHat, UserBias, MovieBias])

    model = Model([UserInput, MovieInput], RattingHat)

    optim = Adam()
    model.compile(optimizer=optim, loss="mse", metrics=[Utility.getRMSE])
    model.summary()
    return model
    
    
def MFDeepNN(intUserSize, intMovieSize, intLatentSize):
    UserInput = Input(shape=(1, ), name="UserInput")
    UserEmbedding = Embedding(intUserSize, intLatentSize, embeddings_initializer="random_normal", embeddings_regularizer=l2(0.001), name="UserEmbedding")(UserInput)
    UserEmbedding = Flatten()(UserEmbedding)
    UserEmbedding = Dropout(0.3)(UserEmbedding)

    MovieInput = Input(shape=(1, ), name="MovieInput")
    MovieEmbedding = Embedding(intMovieSize, intLatentSize, embeddings_initializer="random_normal", embeddings_regularizer=l2(0.001), name="MovieEmbedding")(MovieInput)
    MovieEmbedding = Flatten()(MovieEmbedding)
    MovieEmbedding = Dropout(0.3)(MovieEmbedding)

    hidden = Concatenate()([UserEmbedding, MovieEmbedding])
    hidden = Dense(512, activation="relu")(hidden)
    hidden = Dense(128, activation="relu")(hidden)
    hidden = Dense(32, activation="relu")(hidden)

    RattingHat = Dense(1, activation="relu")(hidden)
    
    model = Model([UserInput, MovieInput], RattingHat)

    optim = Adam()
    model.compile(optimizer=optim, loss="mse", metrics=[Utility.getRMSE])
    model.summary()
    return model