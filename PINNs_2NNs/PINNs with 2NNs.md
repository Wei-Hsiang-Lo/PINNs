---
title: PINNs with 2NNs
tags: [PINNs]

---

# <p class="text-center">PINNs with 2NNs</p>
## Model 1 (Inner neural network)
### Constructure
* Input : $x, y$
* Output : predicted solution $u_1(x,y)$ in the circle
* Hidden layers : There are 6 hidden layers, and 32 neurons in each layer.
* Activation function : hyperbolic tangent - $tanh$
* Optimizer : **Adam** with intial learning rate = $1e^{-2}$
* Boudary condition : $u_1(x, y)$ = $u_2(x, y)$
## Model 2 (Outer neural network)
### Constructure
* Input : $x, y$
* Output : predicted solution $u_2(x,y)$ outside the circle
* Hidden layers : There are 6 hidden layers, and 32 neurons in each layer.
* Activation function : hyperbolic tangent - $tanh$
* Optimizer : **Adam** with intial learning rate = $1e^{-4}$
* Boudary condition : $u_1(x, y)$ = $u_2(x, y)$

### Loss function
$$
L(\theta) = w_1L_{PDE_1}+w_2L_{PDE_2}+w_3L_{Data}
$$
where I choose $w_1,w_2,w_3$ are the weight and, 
$$
L_{PDE_1}(\theta)=\frac{1}{N_c}\sum_{i=1}^{Nc}|\mathcal{F}(u_{1}(x_i,y_i))|^2
$$
$$
L_{PDE_2}(\theta)=\frac{1}{N_2}\sum_{i=1}^{N_2}|\mathcal{F}(u_{2}(x_i,y_i))|^2
$$
$$
L_{Data}=\frac{1}{N_b}\sum_{i=1}^{N_b}|u_1(x_i,y_i)-u_2(x_i, y_i)|^2
$$
$N_c$ = 1000 be the number of collocation points of $NN_1$, $N_2$ = 10000 be the number of collocation points of $NN_2$, and $N_b$ = 200 be the number boundary point of the two neural networks. 
### Data Generation
#### Collocation points of model 1
I choose 1000 points inside the circle with $radius=1$, centered at (0, 0) uniformly. 
And 200 points on the boundary of the circle.
<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/H1WD-vRTgx.png" alt="Figure1" width="80%">
</div>

#### Collocation points of model 2
I choose 10000 points on $\Omega$ = $[-5, 5]\times[-5, 5]\backslash C$, where $C$ denotes the circle with $radius=1$ centered at (0, 0).
<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/SJ1CbwCpeg.png" alt="Figure1" width="80%">
</div>

**The reason why I choose the points in this domian**
Since I use `tf.float64`(IEEE-754 double).
* The maximum number of float **DBL_MAX**$\approx10^{38}$
* We need to make the residual function $\mathcal{F}^2 \leq 10^{38}$
* And consider $u(x,y)=e^{xy}$, by condisdering $y=x$, we have  $|x|\cdot|y|<=26.64$
* So we can only choose $x$ and $y$ on $\Omega$.

### Evaluation
I use $L_1, L_2, L_{inf}$ to evaluate the errors between the approximation function $u_1(x,y)$ and true solution $u_{exact}(x,,y)=e^{xy}$.
Since the integral domain is a circle with $radius=1$
#### Norms 
I replace the form of the norm of function:
$$
\int |f|dxdy = \int |f||J|drd\theta
$$
where $|J|$ denotes the determinant of $Jacobian$ matrix:
$$
|J|=\begin{vmatrix}\frac{\partial{x}}{\partial{r}}&\frac{\partial{x}}{\partial{\theta}} \ \\\frac{\partial{y}}{\partial{r}}&\frac{\partial{y}}{\partial{\theta}}\end{vmatrix}
$$
so the norms are defined as folllow:
$$
||u_1(x,y)-u_{exact}(x,y)||_{L_1}=\int_{0}^{1} |u_1(x,y)-u_{exact}(x,y)||J|drd\theta
$$
$$
||u_1(x,y)-u_{exact}(x,y)||_{L_2}=\sqrt{\int_{0}^{1} |u_1(x,y)-u_{exact}(x,y)|^2|J|drd\theta}
$$
$$
||u_1(x,y)-u_{exact}(x,y)||_{L_{inf}}=max\{{|u_1(x,y)-u_{exact}(x,y)|\}}
$$
#### Numerial Integral
I use Simpson's Rule to calculate the value of integral.
I want to make the accuracy to $O(10^{-8})$.
We can write down the relation between error and intervals of r, intervals of theta:
$$
E\lesssim C_r\frac{R^5}{N_r^4}+C_{\theta}\frac{(2\pi)^5}{N_{\theta}^4}
$$
where $C_r, C_{\theta}$ are relation to the maximum of the fourth derivative of the function.
* Let $u(x,y)=e^{xy}$ and $x,y\in[0,1]$
* we obtain $e^{xy}\leq e^1\approx 2.718$
* The fourth derivatice make it larger $O(10)$
* So, we can roughly estimate $C_r,C_{\theta}\sim 10^{-1}\sim1$
By the relation inquality above, we aasume $C_r, C_{\theta}\sim1, R=1$:
$$
N_r^4\gtrsim\frac{1}{5\times10^{-9}}\approx2\times10^8\Longrightarrow N_r\sim120
$$
$$
N_{\theta}^4\gtrsim\frac{(2\pi)^5}{5\times10^{-9}}\approx7.7\times10^5\Longrightarrow N_{\theta}\sim90
$$
So, I choose $N_r=151$, $N_{\theta}=101$
### Experiment
Then, I will show the traning loss of model with different weight $(w_1,w_2,w_3)$, where $w_1$ is the weight of $L_{PDE_1}$, $w_2$ is the weight of $L_{PDE_2}$, $w_3$ is the weight of L_{Data}
#### $(w_1,w_2, w_3)=(1,1,1)$
<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/BkrBkn0Teg.png" alt="Figure1" width="30%">
</div>
<p class="text-center">Figure 1. Training data loss during 2000 epochs</p>

<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/ryeeG7yCgl.png" alt="Figure1" width="100%">
</div>
<p class="text-center">Figure 2. Loss curves vs. epochs.</p>

#### $(w_1,w_2, w_3)=(0.01,0.01,1)$
<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/ryyw-2Cpee.png" alt="Figure1" width="30%">
</div>
<p class="text-center">Figure 3. Training data loss during 2000 epochs</p>

#### $(w_1,w_2, w_3)=(1,1,0.01)$
<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/SJUSO206le.png" alt="Figure1" width="30%">
</div>
<p class="text-center">Figure 4. Training data loss during 2000 epochs</p>

#### Norms

|  | $(1,1,1)$ | $(0.01,0.01,1)$ | $(1,1,0.01)$ |
| -------- | -------- | -------- | -------- |
| $L_1$ | $2.16588\times10^1$ | $1.42344\times10^{0}$ | $6.18894\times10^0$ |
| $L_2$ | $1.36589\times10^1$ | $8.85012\times10^{-1}$ | $3.51148\times10^0$ |
| $L_{inf}$ | $1.37811\times10^1$ | $1.07916\times10^{0}$ | $2.59614\times10^0$ |