## 1. 作业准备：

**在这个作业中，你将自己用纯python（不掉包！）实现线性回归！**

环境准备:  
           - python 3.6  
           - pycharm education版  
           - 安装 Anaconda  

## 2. 作业介绍：

假如你是一家连锁餐馆的CEO，正在考虑在一个新的街区开一家分店，我们已经开着的店街区人口和利润关联起来，预测新的街区开的店的收入。

具体数据在：[street&profits.txt](https://github.com/dafish-ai/NTU-Machine-learning/blob/master/李宏毅机器学习-作业/week2/street%26profits.txt)

第一列是街区人口（万人）
第二列是获得利润（万元）

我们最终要预测街区**有7万人的时候，利润是多少？**


## 3. 作业提示：

第一步： Part 1:  数据画图 (调用plot_data函数)  
![](https://github.com/dafish-ai/NTU-Machine-learning/blob/master/李宏毅机器学习-作业/week2/pictures/1.1.png)

第二步： Part 2： 梯度下降 Gradient descent (调用gradient_descent)
 
核心知识点：![](https://github.com/dafish-ai/NTU-Machine-learning/blob/master/李宏毅机器学习-作业/week2/pictures/1.2.png)  
          ![](https://github.com/dafish-ai/NTU-Machine-learning/blob/master/李宏毅机器学习-作业/week2/pictures/1.3.png)            ![](https://github.com/dafish-ai/NTU-Machine-learning/blob/master/李宏毅机器学习-作业/week2/pictures/1.4.png)

第三步： Part 3： 预测绘图、结果输出  
 ![](https://github.com/dafish-ai/NTU-Machine-learning/blob/master/李宏毅机器学习-作业/week2/pictures/1.5.png)  
  ![](https://github.com/dafish-ai/NTU-Machine-learning/blob/master/李宏毅机器学习-作业/week2/pictures/1.6.png)

第四步： Part 4： 看懂 Gradient descent 函数代码，自己仿写
