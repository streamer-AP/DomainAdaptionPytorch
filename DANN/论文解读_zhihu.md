# DANN

## 摘要

这篇文章主要提出了一种基于对抗的领域适应方法。主要理论依据是：为了达成好的领域迁移，必须基于一种无法区分源域和目标域的特征进行预测。这个主要处理的场景是源域上的数据带有标记，而目标域上的数据没有标记。在训练的过程中，这个方法的目标是获得：

1. 在源域上对于原任务可区分
2. 关于域之间的偏移不可区分

这篇文章主要通过在标准的分类网络中添加梯度反转层(gradient reversal layer)来实现。并且添加了这种层的神经网络依然可以通过反向传播和随机梯度下降进行训练。在原文章中，作者主要进行了文本情感分析和图像分类的实验，都取得了当时的SOTA的效果。同时作者还进行了行人重识别的实验。

## 原理

许多之前的领域适应都是使用固定的特征表示，领域适应与深度学习相结合，统一到一个训练过程中。文章的整体理论基础基于Ben-David的理论。这一理论认为，给定两个分布，
<img src="https://www.zhihu.com/equation?tex=D_S^X，D_T^X" alt="D_S^X，D_T^X" class="ee_img tr_noresize" eeimg="1">
， 和一个假想的分类函数
<img src="https://www.zhihu.com/equation?tex=H" alt="H" class="ee_img tr_noresize" eeimg="1">
，则这两个分布的
<img src="https://www.zhihu.com/equation?tex=H divergence" alt="H divergence" class="ee_img tr_noresize" eeimg="1">
可以表达为：


<img src="https://www.zhihu.com/equation?tex=d_H(D_S^x,D_T^x)=2sup_{\eta\in H}\left|Pr_{x\sim D_S^x}[\eta(x)=1]-Pr_{x\sim D_T^x}[\eta(x)=1]\right|" alt="d_H(D_S^x,D_T^x)=2sup_{\eta\in H}\left|Pr_{x\sim D_S^x}[\eta(x)=1]-Pr_{x\sim D_T^x}[\eta(x)=1]\right|" class="ee_img tr_noresize" eeimg="1">


如果存在判别函数
<img src="https://www.zhihu.com/equation?tex=\eta" alt="\eta" class="ee_img tr_noresize" eeimg="1">
能够完全区分源域和目标域中的数据，则
<img src="https://www.zhihu.com/equation?tex=H divergence" alt="H divergence" class="ee_img tr_noresize" eeimg="1">
为2.反之，
<img src="https://www.zhihu.com/equation?tex=H" alt="H" class="ee_img tr_noresize" eeimg="1">
中最好的判别函数对源域和目标域区分度越差，则divergence越小。因为普遍情况下我们无法获得全部的源域数据和目标域数据，Ben-David给出了通过采样数据估计divergence的方法，其表达式为：


<img src="https://www.zhihu.com/equation?tex=\hat{d_H}(S,T)=2\left( 1-min_{\eta\in H}\left[\frac{1}{n}\sum_{i=1}^nI[\eta(x_i)=0]+\frac{1}{n'}\sum_{i=n+1}^NI[\eta(x_i)=1]\right]\right)" alt="\hat{d_H}(S,T)=2\left( 1-min_{\eta\in H}\left[\frac{1}{n}\sum_{i=1}^nI[\eta(x_i)=0]+\frac{1}{n'}\sum_{i=n+1}^NI[\eta(x_i)=1]\right]\right)" class="ee_img tr_noresize" eeimg="1">


其中
<img src="https://www.zhihu.com/equation?tex=1-n" alt="1-n" class="ee_img tr_noresize" eeimg="1">
为源域中的数据，
<img src="https://www.zhihu.com/equation?tex=n-N" alt="n-N" class="ee_img tr_noresize" eeimg="1">
为目标域中的数据，即：


<img src="https://www.zhihu.com/equation?tex=U=\left\{ (x_i,0)\right\}_{i=1}^{n}\cup\left\{ (x_i,1)\right\}_{i=n+1}^{N}" alt="U=\left\{ (x_i,0)\right\}_{i=1}^{n}\cup\left\{ (x_i,1)\right\}_{i=n+1}^{N}" class="ee_img tr_noresize" eeimg="1">



<img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1">
为指示器函数，该函数在预测正确时为1，否则为0.即预测准确率越高，divergence越小。

然而实际计算divergence的却困难，因为这涉及到在函数空间内找到一个最优的
<img src="https://www.zhihu.com/equation?tex=\eta" alt="\eta" class="ee_img tr_noresize" eeimg="1">
，所以大多数情况需要使用学习算法。如果使用
<img src="https://www.zhihu.com/equation?tex=\epsilon" alt="\epsilon" class="ee_img tr_noresize" eeimg="1">
表示分类器的错误率，则divergence的估计可以表示为：


<img src="https://www.zhihu.com/equation?tex=\hat{d_A}=2(1-2\epsilon)" alt="\hat{d_A}=2(1-2\epsilon)" class="ee_img tr_noresize" eeimg="1">


Ben-David 同时证明了，
<img src="https://www.zhihu.com/equation?tex=d_H(D_S^x,D_T^x)" alt="d_H(D_S^x,D_T^x)" class="ee_img tr_noresize" eeimg="1">
是
<img src="https://www.zhihu.com/equation?tex=\hat{d_H}(S,T)+O(d)+O(N)" alt="\hat{d_H}(S,T)+O(d)+O(N)" class="ee_img tr_noresize" eeimg="1">
的上界，其中d是函数
<img src="https://www.zhihu.com/equation?tex=H" alt="H" class="ee_img tr_noresize" eeimg="1">
的VC维维度数。因此，在目标域上的风险满足：


<img src="https://www.zhihu.com/equation?tex=R_{D_T}(\eta)\leq R_S(\eta)+\sqrt{\frac{4}{n}(dlog\frac{2en}{d}+log\frac{4}{\delta})}+\hat{d_H}(S,T)+4\sqrt{\frac{1}{n}(dlog\frac{2n}{d}+log\frac{4}{\delta})}+\beta" alt="R_{D_T}(\eta)\leq R_S(\eta)+\sqrt{\frac{4}{n}(dlog\frac{2en}{d}+log\frac{4}{\delta})}+\hat{d_H}(S,T)+4\sqrt{\frac{1}{n}(dlog\frac{2n}{d}+log\frac{4}{\delta})}+\beta" class="ee_img tr_noresize" eeimg="1">


其中

<img src="https://www.zhihu.com/equation?tex=\beta\geq inf_{\eta*\in H}[R_{D_S}(\eta*)+D_{D_T}(\eta*)]" alt="\beta\geq inf_{\eta*\in H}[R_{D_S}(\eta*)+D_{D_T}(\eta*)]" class="ee_img tr_noresize" eeimg="1">

表示分类函数在源域和目标域上分类误差的和的下届。这表明，在给定VC维的情况下，要减少在目标域上的损失，需要最小化

<img src="https://www.zhihu.com/equation?tex=R_S(\eta)+\beta+\hat{d_H}(S,T)" alt="R_S(\eta)+\beta+\hat{d_H}(S,T)" class="ee_img tr_noresize" eeimg="1">
。然而在未知目标域标签的情况下，我们能做的只能是优化

<img src="https://www.zhihu.com/equation?tex=R_S(\eta)+\hat{d_H}(S,T)" alt="R_S(\eta)+\hat{d_H}(S,T)" class="ee_img tr_noresize" eeimg="1">

这篇文章接下来的方法就是构建一种网络去优化上面这个目标函数。

## 网络结构

作者首先考虑了只有一个隐藏层的浅层神经网络，其输入为m维实向量，经过隐藏层后转化为一个D维的表示。即：


<img src="https://www.zhihu.com/equation?tex=G_f(x;W,b)=sigm(Wx+b)" alt="G_f(x;W,b)=sigm(Wx+b)" class="ee_img tr_noresize" eeimg="1">


其中：

<img src="https://www.zhihu.com/equation?tex=sigm(a)=[\frac{1}{1+exp9-a_i}]_{i=1}^{|a|}" alt="sigm(a)=[\frac{1}{1+exp9-a_i}]_{i=1}^{|a|}" class="ee_img tr_noresize" eeimg="1">

在最后的预测层则采用softmax函数进行L分类。即
<img src="https://www.zhihu.com/equation?tex=G_y:R^D\rightarrow [0,1]^L" alt="G_y:R^D\rightarrow [0,1]^L" class="ee_img tr_noresize" eeimg="1">
，使用正确标签的对数可能性作为损失函数，即


<img src="https://www.zhihu.com/equation?tex=L_y(G_y(G_f(x_i)),y_i)=log\frac{1}{G_y(G_f(x))_{y_i}}" alt="L_y(G_y(G_f(x_i)),y_i)=log\frac{1}{G_y(G_f(x))_{y_i}}" class="ee_img tr_noresize" eeimg="1">

最终在网络中进行训练的时候也可以加上正则化项。因为中间隐藏层可以看做是一种表示，所以源域和目标域的样本表示可以分别写作：

<img src="https://www.zhihu.com/equation?tex=S(G_f)=\left\{G_f(x)|x\in S\right\}" alt="S(G_f)=\left\{G_f(x)|x\in S\right\}" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=T(G_f)=\left\{G_f(x)|x\in T\right\}" alt="T(G_f)=\left\{G_f(x)|x\in T\right\}" class="ee_img tr_noresize" eeimg="1">


根据上面的原理，源域和目标域的表示可以写作：

<img src="https://www.zhihu.com/equation?tex=\hat{d_H}(S(G_f),T(G_f))=2\left( 1-min_{\eta\in H}\left[\frac{1}{n}\sum_{i=1}^nI[\eta(G_f(x_i))=0]+\frac{1}{n'}\sum_{i=n+1}^NI[\eta(G_f(x_i))=1]\right]\right)" alt="\hat{d_H}(S(G_f),T(G_f))=2\left( 1-min_{\eta\in H}\left[\frac{1}{n}\sum_{i=1}^nI[\eta(G_f(x_i))=0]+\frac{1}{n'}\sum_{i=n+1}^NI[\eta(G_f(x_i))=1]\right]\right)" class="ee_img tr_noresize" eeimg="1">


在这里为了优化divergence，只需要在网络中添加一个层，这个层来对源域和目标域进行分类，就可以表示
<img src="https://www.zhihu.com/equation?tex=\eta" alt="\eta" class="ee_img tr_noresize" eeimg="1">
.可以使用逻辑回归的形式，损失函数同样也可使用对数损失：

<img src="https://www.zhihu.com/equation?tex=G_d(G_f(x);u,z)=sigm(u^TG_f(x)+z)" alt="G_d(G_f(x);u,z)=sigm(u^TG_f(x)+z)" class="ee_img tr_noresize" eeimg="1">

对于原网络来说，我们的目的是使网络特征表达尽可能差的被域分类器分类，所以原网络的正则化项就可以设计为：

<img src="https://www.zhihu.com/equation?tex=R(W,b)=max_{u,z}-\left[ \frac{1}{n}\sum_{i=1}^{n}L_d^i(W,b,u,z)+\frac{1}{n'}\sum_{i=n+1}^{N}L_d^i(W,b,u,z)\right]" alt="R(W,b)=max_{u,z}-\left[ \frac{1}{n}\sum_{i=1}^{n}L_d^i(W,b,u,z)+\frac{1}{n'}\sum_{i=n+1}^{N}L_d^i(W,b,u,z)\right]" class="ee_img tr_noresize" eeimg="1">

这样，这个正则化项越小，则说明域分类器的最佳表现结果越差。最终目标函数就被导出了：


<img src="https://www.zhihu.com/equation?tex=E(W,V,b,c,u,z)=\frac{1}{n}\sum_{i=1}^{n}L_y^i(W,b,V,c)-\lambda\left( \frac{1}{n}\sum_{i=1}^{n}L_d^i(W,b,u,z)+\frac{1}{n'}\sum_{i=n+1}^{N}L_d^i(W,b,u,z)\right)" alt="E(W,V,b,c,u,z)=\frac{1}{n}\sum_{i=1}^{n}L_y^i(W,b,V,c)-\lambda\left( \frac{1}{n}\sum_{i=1}^{n}L_d^i(W,b,u,z)+\frac{1}{n'}\sum_{i=n+1}^{N}L_d^i(W,b,u,z)\right)" class="ee_img tr_noresize" eeimg="1">


其中属于网络隐藏层和类别预测其的参数
<img src="https://www.zhihu.com/equation?tex=W,V,b,c" alt="W,V,b,c" class="ee_img tr_noresize" eeimg="1">
希望目标函数最大，属于域分类器的参数
<img src="https://www.zhihu.com/equation?tex=u,z" alt="u,z" class="ee_img tr_noresize" eeimg="1">
希望目标函数最小。这两者之间是一个对抗的过程，这也是网络名Domain Adversial Neural Network的来源。这样的目标函数通过常用的反向传播和随机梯度下降是无法训练的，所以作者选择加入一种新的层来解决这个问题。也就是这篇文章的主要技巧GRL。

GRL层表现的很简单，在前向传播时，表现为恒等变换，在反向传播时，将梯度取反。这种函数在实际中不存在，但在神经网络框架中可以很方便的定制。例如在pytorch里面：

``` python
from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
```

## 实验（next)