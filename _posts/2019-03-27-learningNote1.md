---
title: sklearn笔记1-Generalized Linear Models
description: 
categories:
- note
tags:
- sklearn
- machine learning
- supervised learning
---



<!-- more -->

[TOC]

写一下sklearn的学习笔记，算是系统的梳理一下ML的算法原理和使用方法。

机器学习中重要的两个类别是监督学习（supervised learning）和非监督学习(unsupervised learing)，而监督学习中最基础的一种就是线性模型，即目标值是输入变量的线性组合，可以用如下的数学表达式描述：
$$
\hat y(w,x) = w_0 + w_1x_1 + ...+w_px_p
$$
其中${\hat y}$是预测值，${(x_0,x_1,...,x_p)}$是输入变量，在这个模型中，向量${\vec w = ( w_1,...,w_p)}$被定义为系数(coefficient，coef\_)，${w_0}​$ 被定义为截距（intercept\_）。

## 最小二乘法Ordinary Least Squares

线性回归用系数${\vec w}$去拟合一个线性模型，使得在同样的输入变量上，这个线性近似模型的预测值与数据集上的实际值之间的平方差之和最小，数学表达为：
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