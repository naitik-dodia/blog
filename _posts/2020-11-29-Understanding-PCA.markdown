---
layout: post
title: Understanding PCA
date: 2020-11-29 03:29:20 +0530
description: One stop for everything on Principle Component Analysis. # Add post description (optional)
img: pca.png # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Educational, notes]
---

I am writing this article because the material I found for PCA on the internet are a bit too practical. They explain PCA like steps of an algorithm while miss out the mathematical gist behind its working. So this post would make you feel sleepy, as I would be diving deep into the working of PCA. For those who don't know what PCA, this is the complete stop for their reference. And for those who know how to use/implement PCA this would increase their understanding about PCA and the mathematics behind it. Hold tight and don't let your eye lids fall.

# What is PCA ?

Real world datasets are really complex and large. Large in both number of data points and the number of dimensions representing each data point. This is because in machine learning a very simple observation is that the more data your model is trained on the more generic it gets. So when preparing the sample out of the population, beforehand you don't know about how many variables would be enough to capture a datapoint, so you try to capture every feature that you can. This makes the feature set very large but many times these features are not totally different between themselves i.e. many of the features can be derived from some other feature set in the same dataset. If you somehow remove those redundant features then you can reduce the cardinality of the dimensions of your dataset. Reducing the number of features helps a lot as it reduces lot of computation time. PCA is a dimensionality reduction technique. PCA derives new dimensions from the old dimensions which are rich in capturing the variance of the data and linearly independent (any one of these dimensions cannot be derived by any linear combination of the other dimensions)

# Let's dive
Let,
<br>
$$
\begin{align}
m &= number\;of\;datapoints\newline
n &= number\;of\;features\newline
X &= m \times n \; input \; matrix \; of \; dataset \; where\; columns\; represent\; the\; features\; and\; rows\; represent\; the\; datapoints\; \newline
x_i &= i^{th} \; row \; of \; X
\end{align}
$$


The intuition behind PCA is projecting the data on a new basis such that: <br>

- Data has high variance along these vectors. <br>
- The basis vectors are linearly independent  <em>i.e. orthonormal</em> <br>

Let,
<br>
$$
\begin{align}
P &= Projection\; matrix.\;n \times n \; matrix \; whose\; columns \; are\; basis\; vectors (orthonormal)\newline
p_i &= i^{th} \; vector\;of\;P
\end{align}
$$

We want to represent each data point in terms of these vectors.

Basically,
<br>
$$
\begin{align}
x_i &= a_{i1}p_1+a_{i2}p_2+a_{i3}p_3+.....+a_{in}p_n \tag{1}\newline
p_i^ \intercal p_j &= 0\;\;\;\; for \; i \ne j. \;As\;new\;basis\;is\;orthonormal\;and,\newline
p_i^\intercal p_i &= I \;\;\;\; As\;we\;want\;our\;basis\;vectors\;to\;be\;unit\newline
\end{align}
$$

Taking the transpose of equation $(1)$ and having dot product with $p_j$ from right side gives us $a_{ij}$.
<br>
$$
\begin{align}
a_{ij} &= x_i ^\intercal p_j\newline
\therefore \hat{x}_i &= x_i^\intercal P \newline
\therefore \hat{X} &= X ^\intercal P \tag{2}
\end{align}
$$

Now, let us see how to produce covariance matrix $C$ out of data. $C_{ij}$ is the covariance between $i^{th}$ and $j^{th}$ feature in the data. So, C is $n \times n$.
<br>
$$
\begin{align}
C_{ij} &= \frac{1}{m} \sum_{k=1}^{m} (X_{ki} -\mu_i)(X_{kj}-\mu_j) \newline
&= \frac{1}{m} \sum_{k=1}^{m} (X_{ki}X_{kj} -\mu_iX_{kj} - \mu_jX_{ki} + \mu_i\mu_j) \;\;\;\;
Let's\; assume\; the\; data\; has\; zero\; mean\; and\; unit\; variance. \newline
\therefore C_{ij} &= \frac{1}{m} \sum_{k=1}^{m} X_{ki}X_{kj} \newline
C_X &= \frac{1}{m} X^\intercal X \tag{3}
\end{align}
$$

>It is important to note that to arrive at such simplified form we need to <strong>standard normalize</strong> the data. <br>

Also, $X^\intercal X$ is a symmetric matrix $ \because (X^\intercal X)^\intercal = X^\intercal(X^\intercal)^\intercal = X^\intercal X $. In other perspective, we can see that this matrix defines covariance and covariance is commutative, so $C_{ij} = C_{ji}$. <br>

We want the covariance matrix of transformed data so that we can minimize the covariance between the features (make it zero). Let's see if we can get covariance in $\hat{X}$ in this simple form. To get this, we need the $\hat{X}$ to be zero mean. But we already did that by making $X$ zero mean.

Proof: Lets examine feature-wise sum of both $X$ and $\hat{X}$. $1^\intercal X$ gives us feature-wise mean of $X$ and this will be a zero row vector. To Prove: $1^\intercal \hat{X}$ is a zero row vector.
<br>
$$
Eq (2) \implies 1^\intercal \hat{X} = (1^\intercal X) P =0
$$

This allows us to write the covariance matrix of our transformed data as $C_{\hat{X}} = \frac{1}{m}\hat{X}^\intercal \hat{X}$

Ideally, we want,
$$
\begin{align}
(C_{\hat{X}})_{ij} &= 0 \;\;\;\; i\ne j \newline
(C_{\hat{X}})_{ij} &\neq 0 \;\;\;\; i=j
\end{align}
$$

In other words, we want $C_\hat{X}$ to be diagonal.

<br>
$$
\begin{align}
C_\hat{X} &= \frac{1}{m}\hat{X}^\intercal \hat{X} = \frac{1}{m}(XP)^\intercal(XP) = \frac{1}{m}P^\intercal X^\intercal XP \newline
C_\hat{X} &=P^\intercal \Sigma P \;\;\;\; where\; \Sigma =  \frac{1}{m}X^\intercal X \tag{4}
\end{align}
$$

We want $P^\intercal \Sigma P = D \;\; i.e.\; Diagonal\;matrix$ <br>

 To find $P$ and $D$, Let's see Eigen Value Decomposition. <br>


<br> 
## Eigen Value Decomposition

Eigen vectors are special kind of vectors which are very less  affected during transformations (space/basis transformations). When a transformation is applied these vectors only scale but don't change directions.<br><br>
$$
\begin{align}
Av &= \lambda v \\ \tag{4.1}
where, A&:the\;transformation\;applied\\
v &: the \;eigen\;vector\\
\lambda &: constant\;scalar\; |\;Eigen\;value\;of\;v
\end{align}
$$


Each eigen vector is associated with its corresponding eigen value. Writing this for many eigen vectors in matrix form:<br>

<br>
$$
\begin{align}
AV &= A\begin{bmatrix}
\uparrow & \uparrow & & \uparrow \\
v_1 & v_2 & ... & v_n\\
\downarrow & \downarrow & & \downarrow \\
\end{bmatrix}
=
\begin{bmatrix}
\uparrow & \uparrow & \uparrow \\
Av_1 & Av_2 & Av_n\\
\downarrow & \downarrow & \downarrow \\
\end{bmatrix}
\\ \\
&= \begin{bmatrix}
\uparrow & \uparrow & & \uparrow \\
\lambda_1 v_1 & \lambda_2v_2 & ... & \lambda_nv_n\\
\downarrow & \downarrow &  & \downarrow \\
\end{bmatrix}
\\
AV &= V\Lambda ,\; where \; \Lambda=
\begin{bmatrix}
\lambda_1 & 0 & 0 & ... & 0 \\
0 & \lambda_2 & 0 & ... & 0 \\
0 & 0 & \lambda_3 & ... & 0 \\
. & . & . & . & . \\
0 & 0 & 0 & 0 & \lambda_n \tag{4.2} \\
\end{bmatrix}
\end{align}
$$

If  $A^{-1}$ exists, then we can write ([Invertability of matrix](https://mathworld.wolfram.com/InvertibleMatrixTheorem.html)): <br>

<br>
$$
\begin{align}
&A = V \Lambda V^{-1} \;\;\;\; This\;is\;called\;Eigen\;value\;decomposition\;of\;A \tag{4.3}\\
&V^{-1}AV = \Lambda \;\;\;\; This\;is\;called\;Diagonalization\;of\;A
\end{align}
$$


Now let's slowly move to our case. Consider the case when $A$ is square symmetric matrix.

**Theorem: The Eigenvectors of square symmetric matrix are orthogonal** <br>

Proof: <br>
$$
\langle Ax,y \rangle = \langle x,A^\intercal y \rangle
$$


Now, $A$ is symmetric. Let $x$ and $y$ be eigenvectors of $A$ corresponding to distinct eigen values $\lambda$ and $v$. Then,<br>
$$
\lambda \langle x,y \rangle = \langle \lambda x,y \rangle = \langle Ax,y\rangle = \langle x, A^\intercal y \rangle = \langle x, \mu y \rangle = \mu \langle x, y \rangle \\
\begin{align}
\therefore \lambda \langle x, y\rangle &= \mu\langle x,y\rangle \\
(\lambda - \mu)\langle x,y\rangle &= 0 \\
\langle x, y\rangle &= 0 \;\;\;\; (\because \lambda \ne \mu) \\
\therefore x \perp y \\ \tag*{}
\end{align}
$$


Also, consider the case when these eigen vectors are of unit norm. $\|\|x\|\|=1$. <br>

<br>
$$
\begin{align*}
V^\intercal V &= 
\begin{bmatrix}
\leftarrow & v_1 & \rightarrow \\
\leftarrow & v_2 & \rightarrow \\
 & . & \\
 \leftarrow & v_n & \rightarrow
\end{bmatrix} 
\begin{bmatrix}
\uparrow & \uparrow & & \uparrow \\
v_1 & v_2 & ... & v_n \\
\downarrow & \downarrow & & \downarrow
\end{bmatrix}
= I \\
\therefore V^{-1} &= V^\intercal \tag{4.4}
\end{align*}
$$

Eventually our eigen decomposition takes the form of.....  

<font size="5">   
$$
A = V\Lambda V^\intercal \tag{4.5}
$$
</font>

And the diagonalization of A would be 

<font size="5">   
$$
V^\intercal A V = V \tag{4.6}
$$
</font>

Comparing equation $(4)$ and $(4.6)$, we get <br>

<br>
$$
\begin{align}
D &= C_\hat{X} = \Lambda\;,Diagonal\;matrix\;of\;Eigenvalues \\
\Sigma &= \frac{1}{m}X^\intercal X = A\\
P &= V\;, \;\; the\;eigen\;vectors\;of\;A \\ \tag*{}
\end{align}
$$


Let's have the summary so far. We wanted to transform our data to a new basis which has low covariance and low noise. We found out that the new basis would be eigen vectors of $\frac{1}{m}X^\intercal X$. The new basis is orthogonal. As $\frac{1}{m}X^\intercal X$ is symmetric, the eigen vectors are linearly independant. The transformation of data to such new basis is called Principal Component Analysis. Doing this we can inherently lead to dimension reduction i.e. If we find one or more eigen vectors/values to be zero. But we can explicitly reduce the dimension by chosing the eigen vectors along which the variance is the highest (droping low variance eigen vectors). In practise, we choose $top-k$ eigen vectors.

Let's find out which eigen vectors we should drop so that we do not lose much information.

## Reconstruction error

We can represent any vector $x_i$ using our newly found $n$ eigen vectors $P = p_1, p_2, p_3, .... , p_n$ as their linear combination. This is going back to equation $(1)$.<br>

<br>
$$
\begin{align}
x_i &= a_{i1}p_1+a_{i2}p_2+a_{i3}p_3+.....+a_{in}p_n \\
x_i &= \sum_{j=1}^{n}a_{ij}p_j \tag*{}
\end{align}
$$


But we want to select only k vectors, so,<br>

<br>
$$
\hat{x}_i =\sum_{j=1}^{k}a_{ij}p_j \tag*{}
$$


Let's see what error we would be introducing while doing so. We will calculate squared error and try to minimize it.<br>

<br>
$$
\begin{align}
e &= \sum_{i=1}^{m}(x_i - \hat{x}_i)^\intercal (x_i - \hat{x}_i) \\
&= \sum_{i=1}^{m}(\sum_{j=1}^{n}a_{ij}p_j - \sum_{j=1}^{k}a_{ij}p_j)^2 \\
&= \sum_{i=1}^{m}(\sum_{j=k+1}^{n}a_{ij}p_j)^2 \\
&= \sum_{i=1}^{m}(\sum_{j=k+1}^{n}a_{ij}p_j)^\intercal (\sum_{j=k+1}^{n}a_{ij}p_j) \\ \tag*{}
\end{align}
$$


Here, $a_{ij}$ are constants so $p_j$'s are being multiplied to one another. P is formed of eigen vectors which are orthonormal. Eventually, $p_j^\intercal p_j = 1$ and $p_i^\intercal p_j$ for $i \ne j$. <br>

<br>
$$
\begin{align}
e &= \sum_{i=1}^{m}\sum_{j=k+1}^{n} a_{ij}^2 \\
&= \sum_{i=1}^{m}\sum_{j=k+1}^{n} (x_i ^\intercal p_j)^2 ;\;\;\;from\;(2) \\
&= \sum_{i=1}^{m}\sum_{j=k+1}^{n} (x_i ^\intercal p_j)^\intercal (x_i ^\intercal p_j) \\
&= \sum_{i=1}^{m}\sum_{j=k+1}^{n}p_j^\intercal x_i x_i ^\intercal p_j \\
&= \sum_{j=k+1}^{n}p_j^\intercal (\sum_{i=1}^{m} x_i x_i ^\intercal) p_j \\
&= \sum_{j=k+1}^{n}p_j^\intercal (X^\intercal X) p_j \\
&= \sum_{j=k+1}^{n}p_j^\intercal mC p_j; \;\;\;\;\; from\;(3)\tag{5} \\
\end{align}
$$


To minimize this value, let's find the solution to     $\min_{x} x^ \intercal C x$ given the constraint $x^\intercal x=1$. This is a constrained optimization problem so we will use lagrangian multiplier for our solution. <br>

<br>
$$
\begin{align}
L &= x^\intercal Cx - \lambda (x^\intercal x - 1) \\
\frac{\partial L}{\partial x} &= 2Cx - 2\lambda x = 0 \\ 
Cx &= \lambda x \tag{5.2}
\end{align}
$$


What a surprise!!! We didn't even consider $x$ as eigen vector still it turned out to be eigen vector same as our $p$.

Mutiplying equation $(5.2)$ by $x^\intercal$. <br>

<br>
$$
\begin{align}
x^\intercal Cx &= \lambda x^\intercal x \\
&= \lambda \tag*{}
\end{align}
$$


The solution is lies in the extremeties of eigen values. Largest eigen value would give us maximum value and smallest eigen value will give us the minimum value. The solution to equation $(5)$ lies in the eigen vectors with smallest eigen values.

While picking top k vectors, we would first sort the vectors according to their eigen values in descending order and pick the eigen vectors corresponding to the first k eigen values and leave others.

Let's calculate the varaiance of data along this new basis. Taking equation $(2)$ and seeing the $i^th$ dimension of new data. <br><br>
$$
\begin{align}
\hat{X}_i &= X p_i \\
\end{align}
$$


The variance along this dimesion would be, <br><br>
$$
\begin{align}
\frac{1}{m}\hat{X}_i ^ \intercal \hat{X}_i &= \frac{1}{m}p_i^\intercal (X_i ^ \intercal X_i p_i) \\
&= p_i^\intercal \lambda _i p_i \tag*{$\because p_i$ is the eigen vector of $\frac{1}{m}X_i ^ \intercal X_i$} \\
&= \lambda _i \\
\end{align}
$$


Voila!!! The eigen values are the variances of each eigen vector. So, we have done nothing wrong by selecting the vectors having maximum eigen values.

That is all to understand how PCA works. Let's summarize the steps to perform PCA on a data.

1) Standardize the data to zero mean and unit variance.

2) Find covariance matrix $ C = \frac{1}{m} X^\intercal X $.

3) Find the eigen values and eigen vectors of this covariance matrix.

4) Decide the $k$ for selecting top $k$ eigen vectors. You can do it by deciding how much variance you want to keep by deciding a percentage on it. For example find k such that, <br><br>
$$
 \frac{\sum_{i=1}^{k}\lambda _i}{\sum_{i=1}^{n}\lambda _i} \ge 0.95
$$
to keep $95\%$ variance in the data. In this step you will find the number of dimensions being reduced.

5) Create matrix $P$ by aligning these k vectors.

6) Transform the data using $ \hat{X} = X ^\intercal P $.

7) Done



{% highlight python %}
If you want code mention in the comments. I can add it later with an example dataset
{% endhighlight %}
