# Week 6


## Purpose: Text Sentiment Classification

本次作业为 twitter 上收集到的推文，每则推文都会被标注为正面或负面。1：正面、0：负面。希望利用 training dataset 训练一个 RNN 的 model，来预测每个句子所带有的情绪。


## Data 简介
 
* training dataset 分为两个部分。
    * training label data 含有20万个句子，以及每一句所对应到的情绪(注意：每一句都只有一种情绪)，总共有只有两种情绪(0：负面，1:正面)。
    * training unlabel data 含有130万句左右的句子，但并不含有 label，提供做 semi-supervised。

* testing dataset 则是20万句句子。


## Summary

资料处理的部分，先将 training data 句子中的缩写处理成完整的句子，像是 i'm 就会变成 i am, you're 则会变成 you are。

### RNN Model

先将 training data 中的每个字做统计并编码，并取最多数量的前 10000 个字当做字典。

根据我们的字典将句子转换成一串数字，且透过 padding 的方式将每个句子调整成相同长度。

接著透过 embedding layer 将句子中的字转换成向量成为 RNN 的 input。

两层 GRU 使用的 dropout rate 均为 0.5，再接两层 Hidden layer，最后再透 sigmoid 输出预测

### BOW Model

根据前面提到的字典，我们将每个句子传换成跟字典一样维度的向量，并计算字典中每个字出现在句子的个数。以此向量来代表句子。

接著透过 fully connection 的 layer，最后再透过 sigmoid 的输出建构 Bow Model。


### 透过情绪分类不一样但由相同单字所组成的句子来比较 RNN and BOW model

对于 "today is a good day, but it is hot" 这句话情绪为负面，而透过 RNN Model 可以得知模型预测为 0.53547215 稍微偏向正面。

而 "today is hot, but it is a good day" 这句话情绪为正面，而 RNN Model 预测结果为 0.97617346，有相当的信心为正面。

BOW Model 预测为两具句子都为 0.47XXX 偏向负面。这点可以说明 BOW Model 忽略了单字之间的顺序，加上两个句子单字结构是一样所以预测结果也是一样。

也因为忽略了单字间的顺序，以至于情绪分类效果相对于 RNN Model 逊色一点。


## File Stucture

```
week6 
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