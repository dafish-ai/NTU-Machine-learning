
# Week3作业-PM2.5预测-作业要求

[]{}

## 1.作业内容：  
本次作业的资料是从气象局网站下载的真实观测资料，希望大家利用线性回归linear regression或其他方法预测PM2.5的数值。

## 2.作业数据：  
作业使用丰原站的观测记录，分成trainset跟testset，

trainset是丰原站每个月的前20天所有资料。  
testset是从丰原站剩下的资料中取样出来。  

train.csv：每个月前20天的完整资料。

test_X.csv：从剩下的10天资料中取样出连续的10小时为一笔，前9小时的所有观测数据当作feature，第10小时的PM2.5当作answer。一共取出240笔不重复的test data，请根据feauure预测这240笔的PM2.5。
