
## 和大鱼一起Kaggle吧！
我们此次的任务，就是从 Kaggle 平台的 Titanic 比赛分析哪些人更有可能存活，通过机器学习等工具来预测哪些乘客可以幸免于难。在电影中，拥有绅士风度的英国男人们，让妇女和儿童先逃生，而自己却失去了生命；上流社会享有逃生的优先权，大部分存活了下来，而底层人们甚至被关在船舱中无法逃生。那么真实情况真的是这样吗？让我们在真实数据中找寻答案吧！

### 1. 数据集介绍
Titanic 数据集共分两部分：

 - 训练集（train.csv）
 - 测试集（test.csv）

训练集用来分析和训练机器学习模型；
测试集用来预测，测试模型准确率。
数据的变量列表如下：

变量	定义	含义：
survival	是否生存	0 =否，1 =是 pclass	票务舱	1 = 头等舱，2 = 二等舱,3 = 三等舱
sex	性别
Age	年龄多年	 sibsp	同乘兄弟姐妹/配偶 总数	 parch	同乘父母/孩子 总数	 ticket	票号	 fare	票价
cabin	客舱号	 embarked	登船港口	C =瑟堡，Q =皇后镇，S =南安普敦*

注意：

年龄是官方估计的，如果年龄小于 1 岁，可能也包括小数部分。

文件列表将按照如下方法布置：

├── input
│   ├── train.csv         # 训练集
│   ├── test.csv          # 数据集
├── code
│   ├── Titanic.ipynb     # 程序文件
一定要重视上面的结构！这是 Kaggle 比赛代码通用的习惯，共两个文件夹：input 和 code。数据集放在 input 中，代码放在 code 中。

###2. 项目实战
一个完整的数据科学项目，大概分为以下阶段：

探索性数据分析 EDA（Exploratory Data Analysis），从数据的统计分布等特征，探索数据规律；
数据预处理 Preprocessing，特征工程，数据清洗；
建模 Modeling，建模，调参，预测。
以上步骤并非严格按照顺序进行的，很多时候可能要迭代许多次。比如：

在建模阶段，发现模型效果并不够好，则需要重新跳到预处理步骤进行特征工程，甚至要重新进行 EDA，发现更多规律。
基础数据分析
现在，先导入所有需要用到的库函数：

    import pandas as pd
    import numpy as np
    import random as rnd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import KFold, cross_val_score,GridSearchCV
    from sklearn.ensemble import VotingClassifier
    import warnings
    warnings.filterwarnings("ignore")

    %matplotlib inline

接下来，我们通过 Pandas 导入数据。


    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')
    combine = [train_df, test_df]

输出：

['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']
在分析数据之前，要确定哪些数据是分类数据，哪些数据是数值数据。

数据变量类型分为：Numerical（数值数据）和 Categorical（分类数据）。

使用数值数据可以进行加减乘除、取平均数等等操作；分类数据仅有有限个分类，可以被定义成数字，但不可以对其进行运算。

数值数据细分下去，则分为连续数据和离散数据，连续数据可以是在给定范围中的任意值，离散数据是在指定数字集合中的数。

分类数据分为有序和无序，有序数据是有固定顺序的，不可以被改变。无序数据只是名称上不同，并无先后顺序。

#### 2.1 分类数据

有序分类数据：Pclass（头等舱、二等舱、三等舱）
无序分类数据：Survived（是否存活）、Name（姓名）、Sex（性别）、Ticket（票号）、Cabin（客舱号）、Embarked（登陆港口）
数值数据

连续数值数据：Age（年龄）、Fare（票价）
离散数值数据：SibSp（兄弟姐妹夫妻数）、Parch（父母子女数）
查看数据：

    train_df.head()

输出：

    enter image description here

让我们先宏观地从数据集的基础统计特征，来大体了解一下数据集。

    train_df.describe()

输出：

    PassengerId	Survived	Pclass	Age	SibSp	Parch	Fare
    count	891.000000	891.000000	891.000000	714.000000	891.000000	891.000000	891.000000
    mean	446.000000	0.383838	2.308642	29.699118	0.523008	0.381594	32.204208
    std	257.353842	0.486592	0.836071	14.526497	1.102743	0.806057	49.693429
    min	1.000000	0.000000	1.000000	0.420000	0.000000	0.000000	0.000000
    25%	223.500000	0.000000	2.000000	20.125000	0.000000	0.000000	7.910400
    50%	446.000000	0.000000	3.000000	28.000000	0.000000	0.000000	14.454200
    75%	668.500000	1.000000	3.000000	38.000000	1.000000	0.000000	31.000000
    max	891.000000	1.000000	3.000000	80.000000	8.000000	6.000000	512.329200

以上每行含义

count：非空数据的数量
mean：观测平均值
std：观测标准差
min：最小值
25%：第一四分位数，从小到大排名第 25% 的数值
50%：中位数，从小到大排名第 50% 的数值
75%：第三四分位数，从小到大排名第 75% 的数值
max：最大值
从以上分布中，我们可以得出以下结论：

存活比例约 38%
大部分旅客（>75%）并没有与儿女或父母一起乘船
近 30% 的乘客与兄弟姐妹或夫妻一起乘船
票价差距非常大，大部分（>75%）是 31 块，而极少部分旅客票价高达 512 块之高
分类特征分析
train_df.describe(include=['O'])
输出：

Name	Sex	Ticket	Cabin	Embarked
count	891	891	891	204	889
unique	891	2	681	147	3
top	Lester, Mr. James	male	347082	G6	S
freq	1	577	7	4	644
以上每行含义

count：非空数据的数量
unique：唯一样本的个数，如集合 {a,b,c,a}={a,b,c}，则 unique 为 3
top：众数，也就是最常出现的值
freq：出现的频率
通过以上内容，可以看出：

姓名无重复（name.unique=name.count）；
乘客中男性居多（sex.top=male），占比约 64.8%（sex.freq/sex.count=577/891=0.648）；
从客舱总数（cabin.count）和唯一样本数（cabin.unique）可以得出，有部分人是共同住在一个客舱中。
我们查看各项特征与存活率之间的关系，首先是客舱等级：

    train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    输出：

    Pclass	Survived
    0	1	0.629630
    1	2	0.472826
    2	3	0.242363

可以看出，客舱等级高的（富人），生存率远高于客舱等级低的（穷人）。

查看性别与存活率之间的关系：

    train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    输出：

    Sex
    0	female	0.742038
    1	male	0.188908
    可以看出，女性生存率远高于男性。向绅士们致敬！

查看兄弟姐妹和夫妻数与存活率之间的关系：

    train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    输出：

    SibSp	Survived
    1	1	0.535885
    2	2	0.464286
    0	0	0.345395
    3	3	0.250000
    4	4	0.166667
    5	5	0.000000
    6	8	0.000000

可以猜测，有一两个兄弟姐妹/夫妻同乘的人，互相帮助，生存率更高。但如果同乘的人太多（>2）反倒会相互拖累，生存率甚至比独自一个人还低。

如果你同意上面这段话，那么你可能犯错了。因为这只是马后炮式的猜测。通过统计关系猜测原因是极不靠谱的。记住：相关性并不能推出因果性，数据的相关可以在建模中利用，但不要试图从中推论因果性。

接下来，查看“有父母/子女同乘”与存活率之间的关系：

    train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    输出：

    Parch	Survived
    3	3	0.600000
    1	1	0.550847
    2	2	0.500000
    0	0	0.343658
    5	5	0.200000
    4	4	0.000000
    6	6	0.000000

可以看出，该属性不同数量之间，有明显的区别。

通过以上内容，可以得出以下结论：

客舱等级越高，生存的可能性越高；
女性生存率远高于男性；
客舱等级和姓名可作为特征输入到模型中；
SibSp and Parch 特征比较复杂，应当进一步分析。
数值特征分析
查看年龄与生存率的关系：

    g = sns.FacetGrid(train_df, col='Survived')
    g.map(plt.hist, 'Age', bins=20)
    输出：

    enter image description here

左侧是未生存下来的人数，右侧是生存下来的人数。

可以看出：

婴儿生存率很高；
大部分年轻人（20~30）并未生存下来；
绝大部分乘客在 15~35 岁之间；
年龄有明显区别。
所以我们决定：将年龄特征应用于建模中。

接下来看客舱等级：

    grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()
    输出：

    enter image description here

可以看出：

三等舱乘客数量最多，但大多数都死亡了；
头等舱及二等舱的婴儿，几乎都得救了；
头等舱大部分的人生存了下来；
不同客舱等级和生存情况下，年龄的分布是不同的。
于是我们决定：

将客舱等级用于模型的输入；
根据客舱等级及是否生存，预测年龄，来填补年龄的空值。

#### 2.2  数据预处理
特征删除
首先，我们先思考要删除那些特征：

从上面的分类数据的特征可以看出，票号（Ticket）与仓号（Cabin）的 unique 与 count 数量极为相近，通过具体仓号和票号预测几乎无任何泛化能力，决定删除。
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
特征修改
我们要分析可以利用哪些特征，创建出更好的特征。

可以看出，姓名无重复，似乎本身对预测没有用。但姓名通常包含称号，而称号通常代表了阶级，我们建立一个称号特征 Title。姓名的格式都是“称号.名字”，所以我们可以通过正则表达式来把称号提取出来。

    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    pd.crosstab(train_df['Title'], train_df['Sex'])
    输出：

    Sex	female	male
    Title		
    Capt	0	1
    Col	0	2
    Countess	1	0
    Don	0	1
    Dr	1	6
    Jonkheer	0	1
    Lady	1	0
    Major	0	2
    Master	0	40
    Miss	182	0
    Mlle	2	0
    Mme	1	0
    Mr	0	517
    Mrs	125	0
    Ms	1	0
    Rev	0	6
    Sir	0	1

如果直接使用 Title 作为分类数据，是不适合的，因为有很多类，比如 Don、Jonkheer 等。其中包含的样本很少，可能是人为的误输入。所以我们可以将不常见的称号分为一类：

    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
         'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
    输出：

    Title	Survived
    0	Master	0.575000
    1	Miss	0.702703
    2	Mr	0.156673
    3	Mrs	0.793651
    4	Rare	0.347826
    之后，用数字代替 Title 特征的字符串。

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
    现在，就可以将姓名属性删除了。

    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]

接下来，将性别数据按照同样的方法，映射为 0、1 数值，而非字符串。

    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

数据填充
从之前分析阶段，可以看出，年龄、目的地和票价是有空值的。

年龄的分布与客舱等级及是否生存相关，于是，我们之前决定了使用客舱等级来预测年龄，填充年龄的空值。

    guess_ages = np.zeros((2,3))

    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & \
                                      (dataset['Pclass'] == j+1)]['Age'].dropna()

                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                        'Age'] = guess_ages[i,j]

        dataset['Age'] = dataset['Age'].astype(int)
    目的地不好判断，于是我们可以用众数来代替：

    freq_port = train_df.Embarked.dropna().mode()[0]

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    票价用中位数代替：

    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    之前的分析可以看出，不同年龄段的生存率是不同的，但是呈分段形式而非线性。所以我们尝试将年龄分段，而非直接当成数值数据处理。

    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
    输出：

    AgeBand	Survived
    0	(0.34, 16.336]	0.550000
    1	(16.336, 32.252]	0.369942
    2	(32.252, 48.168]	0.404255
    3	(48.168, 64.084]	0.434783
    4	(64.084, 80.0]	0.090909
    可以看出，不同年龄段的生存率的确有差异，所以我们用年龄段代替数值年龄：

    for dataset in combine:    
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age']
    train_df = train_df.drop(['AgeBand'], axis=1)
    combine = [train_df, test_df]

#### 2.3 特征创建
我们可以通过结合 SibSp（同乘兄弟姐妹及夫妻），以及（Parch）父母子女数来创建一个特征家人数，来代表同乘家人的数量。

    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    基于此，我们可以创建另一个特征，是否单独乘船。

    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

接下来，删除多余特征（其实同乘家人数与是否单独乘船虽相关，但不一定完全冲突，你也可以在自己的项目中尝试将两个特征都留下）。

    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]

在分析阶段，我们也发现了生存率随着年龄和客舱等级的不同，有着不同的分布，所以我们尝试将年龄和客舱等级结合，创建一个新特征，年龄 * 客舱等级：

    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass
    将字符串特征 Embarked（目的地）转换为数值数据：

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    将票价也变成区间：

    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

    for dataset in combine:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)
    combine = [train_df, test_df]

#### 2.4 建模
在建模阶段，有很多算法可以选择，而我们此次的任务目标是分类，且是有监督问题，如下算法均可使用：

逻辑斯蒂回归 Logistic Regression
K 近邻 KNN
支持向量机 SVM
朴素贝叶斯分类器 Naive Bayes Classifier
决策树 Decision Tree
随机森林 Random Forrest
感知机 Perceptron

对模型结果的判断，是基于 K 折交叉验证得到的分数，这个分数相对更能说明模型的泛化能力。

所用到的工具是：sklearn.model_selection.cross_val_score。

这个工具的使用方法很简单，它需要传入 4 个参数，按顺序分别是：

Estimator 训练好的分类器
X 需要预测的数据
Y 数据对应的标签
CV 交叉验证生成器（sklearn.model_selection.KFold，KFold 的主要参数为 n_splits，也就是将数据分成多少份）
首先，我们先创建训练集与测试集。

    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test  = test_df.drop("PassengerId", axis=1).copy()

尝试一下逻辑斯蒂回归 Logistic Regression。

逻辑斯蒂回归，其默认各个特征是独立且无关的，通常用作模型的 Baseline，且可以用其作为观测各个特征与预测值间的相关性。

    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    acc_log
    输出：

    80.36

对于逻辑斯蒂回归，我们并没有使用交叉验证。逻辑斯蒂回归通常作为基线算法，主要是用来查看各特征预测时与目标的相关程度。

接下来我们查看逻辑斯蒂回归中，各参数的相关性参数。


    coeff_df = pd.DataFrame(train_df.columns.delete(0))
    coeff_df.columns = ['Feature']
    coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

    coeff_df.sort_values(by='Correlation', ascending=False)
    输出：

    -	Feature	Correlation
    1	Sex	2.201527
    5	Title	0.398234
    2	Age	0.287164
    4	Embarked	0.261762
    6	IsAlone	0.129140
    3	Fare	-0.085150
    7	Age*Class	-0.311199
    0	Pclass	-0.749006

可以看出，称号 Title 是最大的正相关参数，而客舱等级 Pclass 是最大的负相关参数，而我们手工创建的 Age*Class 还是很有用的。

正式建模与调参
一个机器学习模型，通常会有很多参数，参数对模型很重要。调参虽然看似是体力活式的遍历搜索，其实却是一门需要灵（yun）感（qi）的手艺，这是无法用简短的文字就教明白的，所以下面的调参并不复杂，大家可以自己增加更多的参数，找到更好的模型。

此次调参使用到的工具是网格搜索（Grid Search），网格搜索通过遍历给定的参数列表，从而选择最优的参数。

比如对支持向量机分类器进行调参，就要改变其参数。

C：惩罚因子（惩罚因子越高，损失函数对误分类更敏感，准确率更高；相反，则泛化能力较强）；
Kernel：核函数，包括线性核、多项式核、RBF 核等。
等等诸多参数，大家尽可以自己尝试。

使用网格搜索（sklearn.model_selection.GridSearchCV）需要提供两个参数：

Estimator，需要调参的模型，比如支持向量机分类器 SVC；
param_grid，参数列表，字典形式，比如 {'C':[1,2,3],'kernel'=['poly','rbf']}，为建立 6 个支持向量机，参数分别为 [C=1,'poly'],[C=1,'rbf'],[C=2,'poly'],[C=2,'rbf'],[C=3,'poly'],[C=3,'rbf']，分别训练这 6 个模型，最终选出最优的模型。
下面，就使用网格搜索，对 SVM 进行调参（为了运行速度，并没有加入很多候选参数，大家可以根据自己电脑配置，尝试加入更多参数，找到更优的模型）。

    svc = SVC()

    param_grid={'C':[0.45, 0.5, 1]}
    grid_svc = GridSearchCV(estimator=svc,param_grid=param_grid)
    grid_svc.fit(X_train,Y_train)
    print(grid_svc.best_params_)

    Y_pred = grid_svc.predict(X_test)
    acc_svc = round(grid_svc.score(X_train, Y_train) * 100, 2)
    acc_svc
    输出：

    {'C': 0.5}

    83.05

看来支持向量机的准确率要略高于逻辑斯蒂回归。

那么 KNN 呢？

    knn = KNeighborsClassifier(n_neighbors = 3)
    param_grid={'n_neighbors':[4,5,6],'weights':['uniform','distance'],'algorithm':['auto','ball_tree','kd_tree','brute']}
    grid_knn = GridSearchCV(estimator=knn,param_grid=param_grid)
    grid_knn.fit(X_train,Y_train)
    print(grid_knn.best_params_)

    Y_pred = grid_knn.predict(X_test)
    acc_knn = round(grid_knn.score(X_train, Y_train) * 100, 2)
    acc_knn
    输出：

    {'algorithm': 'ball_tree', 'n_neighbors': 5, 'weights': 'uniform'}
    83.73
    接下来尝试朴素贝叶斯：

    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
    acc_gaussian
    输出：

    72.28
    接下来尝试感知机：

    perceptron = Perceptron()
    param_grid={'penalty':[None,'l2','l1','elasticnet'],'eta0':[1,1.2,1.4,1.5]}
    grid_perceptron = GridSearchCV(estimator=perceptron,param_grid=param_grid)
    grid_perceptron.fit(X_train,Y_train)
    print(grid_perceptron.best_params_)

    Y_pred = grid_perceptron.predict(X_test)
    acc_perceptron = round(grid_perceptron.score(X_train, Y_train) * 100, 2)
    acc_perceptron
    输出：

    {'eta0': 1.4, 'penalty': 'l2'}

    78.0
    试试线性 SVM：

    linear_svc = LinearSVC()
    param_grid={'loss':['hinge','squared_hinge'],'C':[0.3,0.5,0.7,1]}
    grid_linear_svc = GridSearchCV(estimator=linear_svc,param_grid=param_grid)
    grid_linear_svc.fit(X_train,Y_train)
    print(grid_linear_svc.best_params_)

    Y_pred = grid_linear_svc.predict(X_test)
    acc_linear_svc = round(grid_linear_svc.score(X_train, Y_train) * 100, 2)
    acc_linear_svc
    输出：

    {'C': 1, 'loss': 'squared_hinge'}
    79.12
    试试随机梯度下降 SGD：

    sgd = SGDClassifier()
    sgd.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
    acc_sgd
    输出：

    63.52
    试试决策树：

    decision_tree = DecisionTreeClassifier()
    param_grid={'splitter':['best','random'],'max_features':['auto', 'sqrt', 'log2', None]}
    grid_decision_tree = GridSearchCV(estimator=decision_tree,param_grid=param_grid)
    grid_decision_tree.fit(X_train,Y_train)
    print(grid_decision_tree.best_params_)

    Y_pred = grid_decision_tree.predict(X_test)
    acc_decision_tree = round(grid_decision_tree.score(X_train, Y_train) * 100, 2)
    acc_decision_tree
    输出：

    {'max_features': None, 'splitter': 'random'}
    86.76
    试试随机森林：

    random_forest = RandomForestClassifier(n_estimators=100)
    param_grid={'n_estimators':[10,15,20,25,30],'max_features':['auto','sqrt','log2',None]}
    grid_random_forest = GridSearchCV(estimator=random_forest,param_grid=param_grid)
    grid_random_forest.fit(X_train,Y_train)
    print(grid_random_forest.best_params_)

    Y_pred = grid_random_forest.predict(X_test)
    acc_random_forest = round(grid_random_forest.score(X_train, Y_train) * 100, 2)
    acc_random_forest
    输出：

    {'max_features': 'auto', 'n_estimators': 25}
    86.76
    接下来，根据以上得到的参数使用投票算法，来进行最后的融合。

    models = []
    kfold = KFold(n_splits=5,random_state=5)
    models.append(('logreg',logreg))
    models.append(('grid_svc',grid_svc))
    models.append(('grid_knn',grid_knn))
    models.append(('gaussian',gaussian))
    models.append(('grid_perceptron',grid_perceptron))
    models.append(('grid_linear_svc',grid_linear_svc))
    models.append(('sgd',sgd))
    models.append(('grid_decision_tree',grid_decision_tree))
    models.append(('grid_random_forest',grid_random_forest))
    ensemble_model = VotingClassifier(estimators=models)

    cross_val_score(ensemble_model, X_train, Y_train, cv=kfold)

    ensemble_model.fit(X_train,Y_train)
    acc_ensemble_model = round(ensemble_model.score(X_train, Y_train) * 100, 2)
    acc_ensemble_model
    输出：

    82.94
    最后，对测试数据进行预测，并导出数据，提交到 Kaggle（需注册账号）。

    submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived": ensemble_model.predict(X_test)
        })
    submission.to_csv('../submission.csv', index=False)

接下来就是你自己要做的事情了，将结果提交到 Kaggle 上，并查看准确率。准确率通常为 79%~80% 之间，在近万名选手中，排名约为前 21%。

这个准确率似乎并不让人满意，原因如下：

数据本身原因，数据本身不够多，且数据表达能力约 79%，所以准确率上限约 79% 是正常的；
不够精细的数据调参，如果您想尝试优化，可以在网格搜索中放入更多的参数。
