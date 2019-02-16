## Week6作业 - 面部情绪分类Image Sentiment Classification
本次作业为网络上收集到的人脸表情资料。经过特殊处理，每张图片均是人脸部分占大部分。
### 1. 数据说明
只需 - 训练集数据文件train.csv

[下载地址1](https://www.kaggle.com/c/7573/download-all)  或 [下载地址2](https://pan.baidu.com/s/1thsXYH-ZH79XqVX6bUlRFw)

约 28000 张图片样本，每行包含label和 feature两个字段，两个字段的说明如下：
> label：每张图片都唯一属于一种表情。共有七种可能的表情（0：生气，1：厌恶，2：恐怖， 3：高兴， 4：难过， 5：惊讶， 6：中立(难以区分为前六种表情))；

```python
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
```

> feature：48*48个灰度强度值。每个数值的取值范围为0-255，其中0为黑，255为白。

提示：使用numpy.reshape函数

### 2. 作业要求 - 代码
要求：数据切分方式 2-8分，随机种子2019。数据划分代码参考如下：
```python
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2019)
```
#### 2.1 构建CNN模型
构建CNN模型，尽可能好地调参（评价指标用准确度）

代码包含：CNN模型架构、训练过程和准确度。

#### 2.2 构建DNN模型
构建DNN模型（和上述CNN模型采用相同参数量），实现本次作业

代码包含：DNN模型架构、训练过程和准确度；并和CNN的结果对比分析

#### 2.3 通过混淆矩阵分析结果
代码包含：多分类的confusion matrix，描述哪些class间容易被弄混。

### 3. 其他可选
更多作业参见
[Assignment #3 - Image Sentiment Classification](https://ntumlta.github.io/ML-Assignment3/index.html)