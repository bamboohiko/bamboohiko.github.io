---
title: sklearn笔记1：一般线性模型 Generalized Linear Models
description: 
categories:
- note
- model
tags:
- sklearn
- machine learning
- supervised learning
---

这一篇是对各种一般线性模型的介绍，其中用到的目标函数同样也适用于其他场合。

<!-- more -->

[TOC]

在阅读*sklearn*文档的时候，发现这其实是非常好的机器学习教材，所以打算写一下sklearn的学习笔记，算是系统的梳理一下ML的算法原理和使用方法，主要采取翻译加上个人认知的形式。

机器学习中重要的两个类别是监督学习（supervised learning）和非监督学习(unsupervised learing)，而监督学习中最基础的一种就是线性模型，即目标值是输入变量的线性组合，可以用如下的数学表达式描述：
$$
\hat y(w,x) = w_0 + w_1x_1 + ...+w_px_p
$$
其中${\hat y}$是预测值，${(x_0,x_1,...,x_p)}$是输入变量，在这个模型中，向量${w = ( w_1,...,w_p)}$被定义为系数(coefficient，coef\_)，${w_0}$ 被定义为截距（intercept\_），而我们的目的就是在已知一组$(X,y)$的情况下，对$ w $和$w_0$进行求解。

## 最小二乘法 Ordinary Least Squares

*LinearRegression* 用系数${\vec w}$去拟合一个线性模型，优化函数为最小二乘法，使得在同样的输入变量上，这个线性近似模型的预测值与数据集上的实际值之间的平方差之和最小，数学表达为：
$$
\min_w ||Xw-y||_2^2
$$
在sklearn的监督学习算法模型中，一般流程为

> 1. 初始化模型，定义模型中的各个参数(Class实例化)
> 2. 使用训练集进行训练（fit函数）
> 3. 输入测试集预测结果（predict函数）

而在非监督学校算法模型中，一般会有一个fit_predict函数来替代上面的2，3两步，具体到线性回归类（sklearn.linear_model.LinearRegression）上，下面是官方文档中给出的示例：

```Python
>>> from sklearn import linear_model
>>> reg = linear_model.LinearRegression()
>>> reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
...                                       
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                 normalize=False)
>>> reg.coef_
array([0.5, 0.5])
```

可以看到，代码中首先初始化了一个LinearRegression类，然后用了数据集$X, y$进行拟合，得到拟合好的模型之后，查看了系数向量，当然也可以调用predict(X)计算预测值。

线性回归是一种非常自然的方法，简单有效并且可解释性强，另外，最小二乘法除了在线性回归模型中被作为优化函数之外，也可以使用其他的函数形式拟合数据集。但线性回归也有其缺点，首先模型过于简单，无法拟合高次函数，同时由于线性模型的线性不变性，当样本点具有近似线性相关性时，有效样本点会减少，使模型对数据集上的随机误差变得敏感，因此，如果要使用线性回归模型，可能需要对数据集做一些筛选，如果结果中误差较大的话，需要考虑使用其他高次模型。但无论如何，线性回归仍然是一种在很多场景下有效的模型，不应该应为其简单而被忽视。

时间复杂度：当输入数据集$X.shape = (n,p)$时，时间复杂度为$O(np^2)$

## 岭回归 Ridge Regression

*Ridge*为了解决最小二乘法的问题，在目标函数中加入了L2范数作为惩罚项，最小化岭系数的残差平方和：
$$
\min_w||Xw-y||_2^2 + \alpha||w||_2^2\\
||w||_2 = \sqrt{\sum_{i=1}^p w_i^2}
$$
$\alpha \ge 0$是一个控制收缩率的复杂度参数，越大时在共线情况下系数的健壮性越好：

![Ridge codfficients as a function of the regularization](https://scikit-learn.org/stable/_images/sphx_glr_plot_ridge_path_0011.png)

和*LinearRegression*类相同，*Ridge*使用$(X,y)$作为fit函数输入进行训练，在类初始化时，可以定义参数$\alpha$的值，时间复杂度与*LinearRegression*相同

> **References**
>
> - “Notes on Regularized Least Squares”, Rifkin & Lippert ([technical report](http://cbcl.mit.edu/publications/ps/MIT-CSAIL-TR-2007-025.pdf), [course slides](https://www.mit.edu/~9.520/spring07/Classes/rlsslides.pdf)).

## Lasso

Lasso是估计稀疏系数的线性模型，它倾向于用更少的参数拟合，有效地减少了解析式所依赖参数地数量。因此，Lasso和其变体成为了压缩感知(Compressed sensing)领域地基础，在特定条件下，它能够复原确定的非零权重集。

在数学表达式上，它是一个用L1范数作为正则化项的线性模型，目标函数为：
$$
\min_w {1 \over 2n_{samples}}||Xw-y||_2^2 + \alpha||w||_1\\
||w||_1=\sum_{i=1}^p |w_i|在*Lasso*中，对参数的优化使用坐标下降法（coordinate descent）实现，在*Least Angle Regression*中有另一种实现方式。
$$
由于加入了L1范数，对系数的求解变成了一个非凸优化问题，在*Lasso*中，对参数的优化使用坐标下降法（coordinate descent）实现，在*Least Angle Regression*中有另一种实现方式。（TODO：坐标下降法）同时，由于Lasso适用于稀疏模型的特性，它也能够被用来做特征选择。（TODO：基于L1范数的特征选择），下面的两篇reference解释了sklearn中使用坐标下降法求解的过程，和用于控制收敛的duality gap计算。

> **References**
>
> - “Regularization Path For Generalized linear Models by Coordinate Descent”, Friedman, Hastie & Tibshirani, J Stat Softw, 2010 ([Paper](https://www.jstatsoft.org/article/view/v033i01/v33i01.pdf)).
> - “An Interior-Point Method for Large-Scale L1-Regularized Least Squares,” S. J. Kim, K. Koh, M. Lustig, S. Boyd and D. Gorinevsky, in IEEE Journal of Selected Topics in Signal Processing, 2007 ([Paper](https://web.stanford.edu/~boyd/papers/pdf/l1_ls.pdf))

