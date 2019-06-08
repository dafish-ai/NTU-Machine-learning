from keras.layers import Input, LSTM, Dense, Dropout, GRU
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.models import Model
from keras.optimizers import Adam


def RNN(dictModelPara):

    inputs = Input(shape=(dictModelPara["intSequenceLength"],), dtype="int32")

    VocabEmbedding = Embedding(dictModelPara["intVocabSize"], dictModelPara["intEmbeddingDim"])(inputs)
    
    if dictModelPara["cell"] == "LSTM":
        cell = LSTM(units=dictModelPara["intHiddenSize"], return_sequences=True)(VocabEmbedding)
        cell = Dropout(dictModelPara["floatDropoutRate"])(cell)
        cell = LSTM(units=dictModelPara["intHiddenSize"], return_sequences=False)(cell)

    elif dictModelPara["cell"] == "GRU":
        cell = GRU(units=dictModelPara["intHiddenSize"], return_sequences=False, dropout=dictModelPara["floatDropoutRate"], recurrent_dropout=dictModelPara["floatDropoutRate"])(VocabEmbedding)


    # output = Dense(dictModelPara["intHiddenSize"]//2, kernel_regularizer=regularizers.l2(0.00001), activation="relu")(output)
    output = Dropout(dictModelPara["floatDropoutRate"])(cell)
    output = Dense(1, activation="sigmoid")(output)

    model = Model(inputs=inputs, outputs=output)

    optim = Adam()
    model.compile(optimizer=optim, loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

    # https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47582