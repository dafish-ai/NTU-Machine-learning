# Week7作业-Text Sentiment Classification-作业要求
本次作业为Text Sentiment Classification，提供给各位的数据集是从Twitter上抓下来的文句。

## 1. 数据集
训练数据集分为两个部分。一个部分含有20万个句子，以及每一句所对应到的情绪（注意：每一句都只有一种情绪），总共有只有两种情绪（0：负面，1：正面） 。另一个部分含有130万句左右的句子，但并不含有标签，提供给同学做半监督的部分

测试数据集则是20万句句子，希望同学可以利用训练数据集训练一个RNN的模型，来预测每个句子所带有的情绪，并存在csv档中。
[数据集下载地址](https://pan.baidu.com/s/1ImcuDwueH2Ju0Jdl7Cn-Zg)

## 2. 作业要求-代码实现

### 2.1 数据集预处理
* n't → nt
* 数字处理
* 连续出现的相同字母处理
* 标点符号处理

### 2.2 构建RNN模型
* 说明你的RNN的模型架构，训练过程(epoch，optimizer，loss fuction)，准确率

### 2.3 构建Bow模型
* 说明你的Bow的模型架构，训练过程(epoch，optimizer，loss fuction)，准确率

### 2.4 请比较“有无”包含标点符号两种不同tokenize的方式。
* 比较RNN模型上，两种不同tokenize方式的准确率，并讨论原因

## 3. 作业拓展

### 3.1 请比较 bag of word 与 RNN 两种不同 model 对于"today is a good day, but it is hot"与"today is hot, but it is a good day"这两句的情绪分数，并讨论造成差异的原因 

### 3.2 请描述在你的semi-supervised方法是如何标记label，并比较有无 semi- supervised training 对准确率的影响。