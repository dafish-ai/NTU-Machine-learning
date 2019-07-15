# Homework 4


## Purpose: Text Sentiment Classification

本次作業為 twitter 上收集到的推文，每則推文都會被標注為正面或負面。1：正面、0：負面。希望利用 training dataset 訓練一個 RNN 的 model，來預測每個句子所帶有的情緒。


## Data 簡介
 
* training dataset 分為兩個部分。
    * training label data 含有20萬個句子，以及每一句所對應到的情緒(注意：每一句都只有一種情緒)，總共有只有兩種情緒(0：負面，1:正面)。
    * training unlabel data 含有130萬句左右的句子，但並不含有 label，提供做 semi-supervised。

* testing dataset 則是20萬句句子。


## Summary

資料處理的部分，先將 training data 句子中的縮寫處理成完整的句子，像是 i'm 就會變成 i am, you're 則會變成 you are。

### RNN Model

先將 training data 中的每個字做統計並編碼，並取最多數量的前 10000 個字當做字典。

根據我們的字典將句子轉換成一串數字，且透過 padding 的方式將每個句子調整成相同長度。

接著透過 embedding layer 將句子中的字轉換成向量成為 RNN 的 input。

兩層 GRU 使用的 dropout rate 均為 0.5，再接兩層 Hidden layer，最後再透 sigmoid 輸出預測

### BOW Model

根據前面提到的字典，我們將每個句子傳換成跟字典一樣維度的向量，並計算字典中每個字出現在句子的個數。以此向量來代表句子。

接著透過 fully connection 的 layer，最後再透過 sigmoid 的輸出建構 Bow Model。


### 透過情緒分類不一樣但由相同單字所組成的句子來比較 RNN and BOW model

對於 "today is a good day, but it is hot" 這句話情緒為負面，而透過 RNN Model 可以得知模型預測為 0.53547215 稍微偏向正面。

而 "today is hot, but it is a good day" 這句話情緒為正面，而 RNN Model 預測結果為 0.97617346，有相當的信心為正面。

BOW Model 預測為兩具句子都為 0.47XXX 偏向負面。這點可以說明 BOW Model 忽略了單字之間的順序，加上兩個句子單字結構是一樣所以預測結果也是一樣。

也因為忽略了單字間的順序，以至於情緒分類效果相對於 RNN Model 遜色一點。


## File Stucture

```
HW4
|    README.md
|    main.py
|    Test.py
|
└─── 01-RAWData
|       training_label.txt
|       training_nplabel.txt
|       testing_data.txt
|       sampleSubmission.csv
|
└─── 02-APData
|       TokenizerDictionary
|
└─── 03-Output
|       model.h5
|       log.csv
|       LossAccuracyCurves.png
|       submission.csv
|
└─── Base
|      __init__.py
|      DataProcessing.py
|      Model.py
|      Train.py
|      Predict.py
|      Utility.py
|___
```


## Reference

* [原始課程作業說明](https://docs.google.com/presentation/d/1HnyZowEamy8N4cUT0gY4aoRZuBTluIuoe8uYQdFxhI0/edit#slide=id.p)

* [Deep learning for NLP](https://pageperso.lis-lab.fr/benoit.favre/dl4nlp/tutorials/03-sentiment.pdf)

* [Sequence Classification with LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)

* [keras：LSTM函數詳解](https://blog.csdn.net/jiangpeng59/article/details/77646186)