
# Homework 4


## Purpose: Binary Classification

本次作业是需要从给定的个人资讯，预测此人的年收入是否大于50K。

## Data 简介

* 本次作业使用 [ADULT dataset](https://archive.ics.uci.edu/ml/datasets/Adult)

Barry Becker从1994年的人口普查数据库中进行了提取。 
（（AGE> 16）&&（AGI> 100）&&（AFNLWGT> 1）&&（HRSWK> 0））提取了一组合理清洁的记录。

* 共有32561笔训练资料，16281笔测试资料，其中资料维度为106。


## Summary 总结

本次作业执行 generative model 和 discriminative model。

- generative model 的困难点在于要假设资料的分配和变量之间的关系，如果资料越符合假设的分配效果也就越好
- discriminative model 的难点则是在于如何选择超参数

### Logistic Regression

一般对于二元分类最常用的方法为逻辑式回归(Logistic Regression)，其背后有一些统计的推导过程，在此就不多做说明，简单说逻辑式回归跟一般线性回归差别只在于计算线性回归之后再利用sigmoid函数将数值转换到0~1之间，另外将转换过的的数值透过门槛值来区分类别，而门槛值得设置可以根据资料的不同来做设计，常用门槛值为0.5。

在这作业我们将所有的训练资料中的20%当成验证集，由另外80%的资料集来训练参数。并使用 Mini-batch Gradient Descent 演算法来训练逻辑式回归的参数W和B，门槛值则用最一般的方式设置0.5。由下图可以清楚的看出随著叠代次数越来越多，不论是训练集或是验证集的 Cross entropy 都越来越小，且趋近于一致。这也就说明了模型参数学习得不错。最后在测试集的预测精准度为85%。

![](02-Output/TrainProcess.png)

### Probabilstic Generative Model

由于我们的目标是将资料进行二元分类，可以假设年收入大于50(y=1)为<img src="https://latex.codecogs.com/gif.latex?C_{1}" title="C_{1}" />类别和年收入小于50(y=0)为<img src="https://latex.codecogs.com/gif.latex?C_{2}" title="C_{2}" />类别且各为106维的常态分配，且每个特征是独立的，其中变异数矩阵共用，最后由最大估计法直接计算参数<img src="https://latex.codecogs.com/gif.latex?\mu&space;_{1},&space;\mu&space;_{2},&space;\Sigma" title="\mu _{1}, \mu _{2}, \Sigma" />的最佳解。

拥有了模型的参数，我们藉由机率的方式来决定资料是属于哪个类别，也就是说，分别计算资料来自于第一类的机率<img src="https://latex.codecogs.com/gif.latex?P(C_{1})" title="P(C_{1})" />和第二类的机率<img src="https://latex.codecogs.com/gif.latex?P\left&space;(C_{2}&space;\right&space;)" title="P\left (C_{2} \right )" />以及资料在第一类的机率<img src="https://latex.codecogs.com/gif.latex?P(x\mid&space;C_{1})" title="P(x\mid C_{1})" />和第二类的机率<img src="https://latex.codecogs.com/gif.latex?P(x\mid&space;C_{2})" title="P(x\mid C_{2})" />，最后藉由上述这些机率去计算资料属于第一类的机率<img src="https://latex.codecogs.com/gif.latex?P(x\mid&space;C_{1})=&space;\frac{P(x\mid&space;C_{1})P(C_{1})}{P(x\mid&space;C_{1})P(C_{1})&plus;P(x\mid&space;C_{2})P(C_{2})}" title="P(x\mid C_{1})= \frac{P(x\mid C_{1})P(C_{1})}{P(x\mid C_{1})P(C_{1})+P(x\mid C_{2})P(C_{2})}" />和第二类的机率<img src="https://latex.codecogs.com/gif.latex?1-P(x\mid&space;C_{1})" title="1-P(x\mid C_{1})" />，最后藉此机率决定资料类别。

在此作业我们假设资料来自于常态分配，主要的原因还是因为数学推导相对而言比较简单加上常态分配相对而言比较直观，当然要假设其他机率分配也是可行的，例如像是0和1的类别资料，假设百努力分配相对于常态分配就会比较合理，另外假设每个特徵是独立的也就是使用 Naive Bayes Classifier。

在这 case 底下我们的预测精准度大约76%，相对于 discriminative model 的 Logistic Regression 略差一些。另外我们做了很多的假设，像是资料来自于两个常态分配且变异数矩阵使用相同的参数，以及特徵之间是独立，但可能这些资料并不符合这些假设，这也是这个模型的预测率相对于 Logistic Regression 差的原因。
