## 一、整体流程

![WR-MF](./img/WR-MF.jpg)

## 二、数据加载

我们使用了[LastFM](https://grouplens.org/datasets/hetrec-2011/)数据集，该数据集收集了Last.fm网站上2千名用户的社交网络、tagging和music artist listening信息。

通过`data_loader`函数读取数据并将原始数据构建为*[用户index、物品index、权重]*格式的数据集，一个用户可以对应多个物品，一个物品也可以对应多个用户，权重则表示该物品对该用户的重要程度，或者说用户对物品的喜爱程度。最后返回用户列表、物品列表及数据集。

## 三、拆分数据集

根据加载的三元组数据使用简单交叉验证，首先通过`data_split`函数将数据集打乱并按一定比例划分出训练数据和测试数据，然后再将训练数据分为训练集和验证集，最后调用`create_sparse`函数构建稀疏矩阵。矩阵的列为用户index、物品index，矩阵的值则为一对用户-内容的权重。最后返回训练集稀疏矩阵、验证集稀疏矩阵和测试集稀疏矩阵作为最终的训练集、验证集、测试集。

## 四、模型训练

###1.ALS原理
####问题描述
ALS的矩阵分解算法常应用于推荐系统中，将用户(user)对商品(item)的评分矩阵，分解为用户对商品隐含特征的偏好矩阵，和商品在隐含特征上的映射矩阵。与传统的矩阵分解SVD方法来分解矩阵R($R\in \mathbb{R}^{m\times n}$)不同的是，ALS(alternating least squares)希望找到两个低维矩阵，以 $\tilde{R} = XY$ 来逼近矩阵R，其中 ，$X\in \mathbb{R}^{m\times d}$，$Y\in \mathbb{R}^{d\times n}$，d 表示降维后的维度，一般 d<<r，r表示矩阵 R 的秩，$r<<min(m,n)$。

####目标函数

 - 为了找到低维矩阵$X$,$Y$最大程度地逼近矩分矩阵R，最小化下面的平方误差损失函数。
    $$L(X,Y) = \sum_{u,i}(r_{ui} - x_{u}^{T}y_{i})^{2}......(1)​$$    

 - 为防止过拟合给公式 (1) 加上正则项，公式改下为：
    $$L(X,Y) = \sum_{u,i}(r_{ui} - x_{u}^{T}y_{i})^{2} + \lambda (\left \|  x_{u}\right \|^{2} +　\left \|  y_{i}\right \|^{2})......(2)​$$

    其中$x_{u}\in \mathbb{R}^{d}，y_{i}\in \mathbb{R}^{d}​$，$1\leqslant u\leqslant m​$，$1\leqslant i\leqslant n​$，$\lambda​$是正则项的系数。

####模型求解
 - 固定Y，对$x_{u}$ 求导 $\frac{\partial L(X,Y)}{\partial x_{u}} = 0$，得到求解$x_{u}$的公式

    $$x_{u} = (Y^{T}Y + \lambda I )^{-1}Y^{T}r(u)......(3)$$ 


 - 同理固定X,可得到求解$y_{i}​$的公式

    $$y_{i} = (X^{T}X + \lambda I )^{-1}X^{T}r(i)......(4)$$


    其中，$r_{u}\in \mathbb{R}^{n}​$,$r_{i}\in \mathbb{R}^{m}​$,I表示一个d * d的单位矩阵。

 - 基于公式(3)、(4)，首先随机初始化矩阵X，然后利用公式(3)更新X，接着用公式(4)更新Y，直到迭代次数达到最大迭代次数为止。


###2.WR-MF模型
以上模型适用于用户对商品的有明确的评分矩阵的场景，然而很多情况下用户没有明确的反馈对商品的偏好，而是通过一些行为隐式的反馈。比如对商品的购买次数、对电视节目收看的次数或者时长，这时我们可以推测次数越多，看得时间越长，用户的偏好程度越高，但是对于没有购买或者收看的节目，可能是由于用户不知道有该商品，或者没有途径获取该商品，我们不能确定的推测用户不喜欢该商品。WR-MF通过置信度的权重来解决此问题，对于我们更确信用户偏好的项赋予较大的权重，对于没有反馈的项，赋予较小的权重。模型如下：

- WR-MF目标函数

    $\underset{x_{u},y_{i}}{min} L(X,Y) = \sum_{u,i}c_{ui}(p_{ui} - x_{u}^{T}y_{i})^{2} + \lambda (\left \|  x_{u}\right \|^{2} +　\left \|  y_{i}\right \|^{2})......(5)$

    其中
    $$p_{ui} = 
    \begin{cases}
    & \text{1 if }  r_{ui} > 0 \\ 
    & \text{0 if }  r_{ui} = 0
    \end{cases}​$$

    $c_{ui} = 1 + \alpha r_{ui}$，$\alpha$是置信度系数

- 通过最小二乘法求解

    $$x_{u} = (Y^{T}C^{u}Y + \lambda I )^{-1}Y^{T}C^{u}r(u)......(6)$$

    $$y_{i} = (X^{T}C^{i}X + \lambda I )^{-1}X^{T}C^{i}r(i)......(7)$$

    其中$C^{u}$是一$n\times n$维的个对角矩阵，$C_{ii}^{u} = c_{ui}$; 其中$C^{u}$是一$m\times m$维的个对角矩阵，$C_{ii}^{u} = c_{ui}$

- 基于公式(6)、(7)，首先随机初始化矩阵X，然后利用公式(6)更新X，接着用公式(7)更新Y，直到迭代次数达到最大迭代次数为止。


###3.与其他矩阵分解算法的比较
 - 在实际应用中，由于待分解的矩阵常常是非常稀疏的，与SVD相比，ALS能有效的解决过拟合问题。
 - 基于ALS的矩阵分解的协同过滤算法的可扩展性也优于SVD。
 - 与随机梯度下降的求解方式相比，一般情况下随机梯度下降比ALS速度快；但有两种情况ALS更优于随机梯度下降：
    - 当系统能够并行化时，ALS的扩展性优于随机梯度下降法。
    - WR-MF能够有效的处理用户对商品的隐式反馈的数据。

###4.参考文献
 - [Collaborative Filtering for Implicit Feedback Datasets](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=4781121)

 - [Matrix Factorization Techniques for Recommender Systems](http://rakaposhi.eas.asu.edu/cse494/lsi-for-collab-filtering.pdf)

## 五、模型保存与加载

训练前，使用`save_model`或`save_tf_model`函数判断是否存在保存的模型文件，若存在，则使用`load_model`或`load_tf_model`函数加载模型继续训练。训练时，每训练指定轮数，将模型参数保存到指定路径。

## 六、模型测试与评估

测试时获取验证集用户，对各个用户进行TopN推荐，并使用`Metrics`类中的`ranking_measure`静态方法计算TopN类型推荐算法的指标，包括：MAP、NDCG、召回率、精确率、F1值。