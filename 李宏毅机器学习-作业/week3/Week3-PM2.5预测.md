
# Week3作业-PM2.5预测-作业要求

![enter image description here](https://github.com/dafish-ai/NTU-Machine-learning/blob/master/%E6%9D%8E%E5%AE%8F%E6%AF%85%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-%E4%BD%9C%E4%B8%9A/week3/pic/pm2.5.jpg)  

## 1.作业内容：  
本次作业的资料是从气象局网站下载的真实观测资料，希望大家利用线性回归linear regression或其他方法预测PM2.5的数值。

## 2.作业数据：  
作业使用丰原站的观测记录，分成trainset跟testset，

trainset是丰原站每个月的前20天所有资料。  
![enter image description here](https://github.com/dafish-ai/NTU-Machine-learning/blob/master/%E6%9D%8E%E5%AE%8F%E6%AF%85%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-%E4%BD%9C%E4%B8%9A/week3/pic/train.png) 



testset是从丰原站剩下的资料中取样出来
![enter image description here](https://github.com/dafish-ai/NTU-Machine-learning/blob/master/%E6%9D%8E%E5%AE%8F%E6%AF%85%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-%E4%BD%9C%E4%B8%9A/week3/pic/test.png) 

train.csv：每个月前20天的完整资料。


## 我们要预测

在测试数据集中，从剩下的10天资料中取样出连续的10小时为一笔，前9小时的所有观测数据当作feature，第10小时的PM2.5当作answer。一共取出240笔不重复的test data，请根据feature预测这240笔的PM2.5。

![enter image description here](https://github.com/dafish-ai/NTU-Machine-learning/blob/master/%E6%9D%8E%E5%AE%8F%E6%AF%85%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-%E4%BD%9C%E4%B8%9A/week3/pic/result.png)  


![具体数据和代码](https://github.com/dafish-ai/NTU-Machine-learning/tree/master/%E6%9D%8E%E5%AE%8F%E6%AF%85%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-%E4%BD%9C%E4%B8%9A/week3)


