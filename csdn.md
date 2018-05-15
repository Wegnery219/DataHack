引言

这篇文章主要介绍逻辑回归背后的一些概率概念，给你一些直观感觉关于它的代价函数的由来。并且我也介绍了关于最大似然估计（maximum likelihood）的概念，用这个强大的工具来导出逻辑回归的cost函数。接着，我用scikit-learn训练了感知机模型来让你熟悉scikit-learn，最后用scikit-learn来训练逻辑回归，并作出决策边界图，效果还算不错。
逻辑函数(logistic function)

为了更好地解释逻辑回归，让我们首先了解一下逻辑函数。逻辑函数由于它的S形，有时也被称为sigmoid函数。

现在我要引入比值比（odds ratio）的概念，它可以被写成p(1−p)
，其中的p

代表正事件（positive event）的概率，正事件并不是代表好的方面的概率，而是代表我们想要预测的事件。比如：病人患有某种疾病的概率。我们把正事件的类标签设置为1。比值比的对数称为Logit函数，它可以写成如下形式：

logit(p)=logp(1−p)

它的函数图像如下：

Logit函数

图片来源 https://en.wikipedia.org/wiki/Logit#/media/File:Logit.svg

从图像上我们可以看出，logit函数输入0到1的值并把它们转换为整个实数范围内的值。上面的p
代表正事件的概率，因此在给定特征向量x的条件下，类别y=1的概率可以写成P(y=1|x)。大家都知道概率的范围为0到1，如果我把这个概率传递给logit函数那么它的输出范围是整个实数，因此如果我用某些合适的权重向量w参数化特征向量x

后，可以有如下等式：

logit(P(y=1|x))=w0x0+w1x1+⋯+wnxn=∑i=0nwixi

但是在实际应用中，我们更想求的是P(y=1|x)
，因此，我们需要找到logit函数的反函数，通过输入用权重向量w来参数化的x，来输出我们想要知道的正事件的概率，即P(y=1|x)。而这个反函数就是我们的sigmoid函数，它可以写成S(h)=11+e−h，公式中的h为样本特征和权重的线性组合，即，w0x0+w1x1+⋯+wnxn

。下面我们来画出这个函数图像的样子：

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(h):
    return 1.0 / (1.0 + np.exp(-h))

h = np.arange(-10, 10, 0.1) # 定义x的范围，像素为0.1
s_h = sigmoid(h) # sigmoid为上面定义的函数
plt.plot(h, s_h)
plt.axvline(0.0, color='k') # 在坐标轴上加一条竖直的线，0.0为竖直线在坐标轴上的位置
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted') # 加水平间距通过坐标轴
plt.axhline(y=0.5, ls='dotted', color='k') # 加水线通过坐标轴
plt.yticks([0.0, 0.5, 1.0]) # 加y轴刻度
plt.ylim(-0.1, 1.1) # 加y轴范围
plt.xlabel('h')
plt.ylabel('$S(h)$')
plt.show()

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17

sigmoid函数

从上图我们可以看出，函数接收整个实数范围的输入，输出0到1之间的数。

因此S(w0x0+w1x1+⋯+wnxn)=P(y=1|x;w)
，这个概率我们可以解释成：给定用权重w参数化后的特征x

，样本属于类别1的概率。通过阶跃函数（step function），我们可以得到如下公式：

f(n)={1,0,if S(h) ≥0.5otherwise

还有一个等价的公式如下：

f(n)={1,0,if h ≥0.0otherwise

实际上，很多应用不只仅仅是想得到一个类标签，而是算出属于某个类别的概率。比如逻辑回归就是这样的，它不仅仅是告诉你是否患有疾病，而是告诉你有多大概率患有这个疾病。

在上面的例子当中，我们一直都看到权重w
的出现，那么我们如何学习出最佳的权重w

呢？在告诉你答案之前，让我们先复习一下最大似然估计（maximum likelihood）的概念。
最大似然估计（maximum likelihood）

这个方法的本质就是：选择最佳的参数值w

，来最大化我们样本数据的可能性。

假设我们给定样本X1,X2,X3,…,Xn
，那么我可以写出一个关于参数w

的可能性函数，如下：

lik(w)=f(X1,X2,X3,…,Xn|w)

实际上，可能性函数就是样本数据作为参数w

的函数的概率。

如果X1,X2,X3,…,Xn

相互之间是独立的，可能性函数可以简化成如下形式：

lik(w)=∏1nf(Xi|w)

但是，如果我们有很多的样本数据呢？这时，你就会乘上很多项，这些项通常都很小，可能性函数就会变得很小。因此，你应该采用log可能性函数。第一，如果在可能性很小的时候它可以防止潜在的数值下溢;第二，我们把乘积转换为求和，这可以使我们更加容易求得函数的导数。第三，log函数是单调的，最大化可能性函数的值也就是最大化log可能性函数的值。log可能性函数公式如下：

l(w)=log(lik(w))=∑inlog(f(Xi|w))

下面，我举2个例子来应用一下这个强大的工具：

1、假设你口袋里有2枚硬币，1枚硬币出现正面的概率为p=0.5
，另1枚硬币出现正面的概率为p=0.8

，现在你从口袋里随机拿出一枚硬币（你并不知道拿的是哪枚硬币），然后随机投掷4次，出现3次正面，1次反面。你觉得你拿出的是哪枚硬币？或者哪枚硬币的最大似然估计更大？

    答：通过问题我们可以得出这个是属于二项分布。它的概率为P(x|n,p)=(nx)px(1−p)n−x

.现在，我们来写出log可能性函数：
l(p)=log((43)p3(1−p))

由于我们已经给出了p的值只能为0.5或0.8，因此，我们不必求导来算出p的值最大化可能性。这里我们只需要把两个p值代入就行了，分别得出如下结果：
l(0.5)=−0.6021
l(0.8)=−0.3876
因此当p为0.8时，使可能性函数更大，所以我更可能拿出的是正面概率为p=0.8

    的硬币。

2、假设Xi∼N(μ,σ2)

，并且相互之间是独立的。求出最佳参数？

    答：log可能性函数如下：
    l(μ,σ2)=∑i=1nlog[1σ2π−−√exp(−(Xi−μ2)2σ2)]=−∑i=1nlog(σ)−∑i=1nlog(2π−−√)−∑i=1n[(Xi−μ)22σ2]=−nlog(σ)−nlog(2π−−√)−12σ2∑i=1n(Xi−μ)2


因为我们想找到参数μ和σ使得可能性函数最大，因此我们需要找到它们的偏导：
∂l(μ,σ2)∂μ=∂∂μ(−nlog(σ)−nlog(2π−−√)−12σ2∑i=1n(Xi−μ)2)=−1σ2∑i=1n(Xi−μ)

∂l(μ,σ2)∂σ2=∂∂σ2(−n2log(σ2)−nlog(2π−−√)−12(σ2)−1∑i=1n(Xi−μ)2)=−n2σ2+12(σ2)−2∑i=1n(Xi−μ)2

让两个偏导都等于0,然后求出最佳参数。
μ=1n∑i=1nXi=X¯

σ2=1n∑i=1n(Xi−μ)2

掌握了最大似然估计，现在你就可以知道逻辑回归cost函数的由来了。
逻辑回归的cost函数

现在，我们可以用可能性函数来定义上面的权重w

了。公式如下：

L(w)=∏i=1nP(y(i)|x(i);w)=∏i=1nS(h(i))y(i)(1−S(h(i)))1−y(i)

上面公式中的h为假设函数w0x0+w1x1+⋯+wnxn

，把上面的函数加上对数，公式如下：

l(w)=log(L(w))=∑i=1ny(i)log(S(h(i)))+(1−y(i))log(1−S(h(i)))

现在，我们的目的是最大化log可能性函数，找到一个最佳的权重w

。我们可以在上面的log可能性函数前加上负号，用梯度下降算法来最小化这个函数。现在，我得到了逻辑回归的cost函数如下：

J(w)=−∑i=1ny(i)log(S(h(i)))+(1−y(i))log(1−S(h(i)))

    n

：训练集样本总数
S
：sigmoid函数
h

    ：假设函数

假设我们只有一个样本，现在我们可以把上面的cost函数拆分成分段函数如下：

J(w)={−log(S(h)),−log(1−S(h)),if y = 1if y = 0

Logistic regression cost function

我们把逻辑回归的cost函数做成了图像如上。当实际类别为1时，如果我们预测为1则需要很小的cost，如果我们预测为0则需要很大的cost;反过来，当实际类别为0时，如果我们预测为0则需要很小的cost，如果我们预测为1则需要很大的cost

下面是一些关于逻辑回归更详细的一些理论知识，感兴趣的可以看看。

http://czep.net/stat/mlelr.pdf

下面我要用scikit-learn的逻辑回归来拟合Iris数据集。
Iris数据集概述

首先，我们取得数据，下面这个链接中有数据的详细介绍，并可以下载数据集。

https://archive.ics.uci.edu/ml/datasets/Iris

从数据的说明上，我们可以看到Iris有4个特征，3个类别。但是，我们为了数据的可视化，我们只保留2个特征（sepal length和petal length）。让我们先看看数据集的散点图吧！

下面，我们进入Ipython命令行。

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None) # 加载Iris数据集作为DataFrame对象
X = df.iloc[:, [0, 2]].values # 取出2个特征，并把它们用Numpy数组表示

plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='setosa') # 前50个样本的散点图
plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='x', label='versicolor') # 中间50个样本的散点图
plt.scatter(X[100:, 0], X[100:, 1],color='green', marker='+', label='Virginica') # 后50个样本的散点图
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc=2) # 把说明放在左上角，具体请参考官方文档
plt.show()

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14

Iris数据集实例

从上图我们可以看出，数据集是线性可分的。为了更好地演示效果，我只用setosa和versicolor这两个类别。

接下来，我们要用强大的scikit-learn来拟合模型。
初识scikit-learn

有一些开源库已经把机器学习算法封装到黒盒中，别人已经替我们做了大量的工作。在实际应用中，我们主要的工作是对数据的预处理、挑选出好的特征、调试算法、多试一些算法，比较它们之间的性能、选择出好的模型。因此我并不会建议你自己去实现这些机器学习算法。那么你可能会有疑问，既然别人已经都实现了，我们还了解这些算法有什么用？就和我刚才说的一样，我们大部分的工作都是在调试算法，找到其最好的性能，如果你不了解它们的原理，你能知道怎么调试它们吗？

现在我要用scikit-learn来训练感知机模型，让你了解一下scikit-learn的一些方法。如果你并不了解感知机模型（perceptron），你可以去Google一下，有很多关于它的文章，这个算法很早就出现了，也很简单。

在用scikit-learn之前，你需要安装它，步骤请参考http://scikit-learn.org/stable/install.html

下面，让我们来看看scikit-learn的强大吧！

from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris() # 由于Iris是很有名的数据集，scikit-learn已经原生自带了。
X = iris.data[:, [2, 3]]
y = iris.target # 标签已经转换成0，1，2了
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # 为了看模型在没有见过数据集上的表现，随机拿出数据集中30%的部分做测试

# 为了追求机器学习和最优化算法的最佳性能，我们将特征缩放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train) # 估算每个特征的平均值和标准差
sc.mean_ # 查看特征的平均值，由于Iris我们只用了两个特征，所以结果是array([ 3.82857143,  1.22666667])
sc.scale_ # 查看特征的标准差，这个结果是array([ 1.79595918,  0.77769705])
X_train_std = sc.transform(X_train)
# 注意：这里我们要用同样的参数来标准化测试集，使得测试集和训练集之间有可比性
X_test_std = sc.transform(X_test)

# 训练感知机模型
from sklearn.linear_model import Perceptron
# n_iter：可以理解成梯度下降中迭代的次数
# eta0：可以理解成梯度下降中的学习率
# random_state：设置随机种子的，为了每次迭代都有相同的训练集顺序
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

# 分类测试集，这将返回一个测试结果的数组
y_pred = ppn.predict(X_test_std)
# 计算模型在测试集上的准确性，我的结果为0.9，还不错
accuracy_score(y_test, y_pred)

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    23
    24
    25
    26
    27
    28
    29
    30
    31
    32

注意：如果上面的学习率过大，算法可能会越过全局最小值，收敛地很慢，甚至有可能发散。如果学习率过小，算法需要很多次迭代才能收敛，这会使学习过程很漫长。

现在估计你已经大概了解scikit-learn了，接下来，我们来实现逻辑回归。
scikit-learn实现逻辑回归

scikit-learn的逻辑回归中有很多调节的参数，如果你不太熟悉，请参考下面的文档。

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

下面，我将用scikit-learn来训练一个逻辑回归模型：

from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
lr.predict_proba(X_test_std[0,:]) # 查看第一个测试样本属于各个类别的概率
plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    23
    24
    25
    26
    27

scikit-learn实现逻辑回归

从上图我们看到逻辑回归模型把类别很好地分开了。还有一点你需要注意的是，上面代码中的plot_decision_regions函数是我自己写的，scikit-learn中并没有这个函数，我已经把这个函数上传到Github