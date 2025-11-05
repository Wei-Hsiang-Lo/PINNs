---
title: Training Performance of different decay rate function
tags: [PINNs]

---

# <p class="text-center">Performance of Different Decay Rate</p>
## Problem
Using the same constructure of DNN model to the training performance on fast dacay function and slow decay function.
The functions are defined as below:
* **fast**: $u_1(x,y)=e^{-(2x^2+3y^2)}$
* **slow**: $u_2(x,y)=\frac{1}{x^2+y^2+1}$

In the following, the accuracy of numerical integral is $O(10^{-8})$. 
So, I only choose the domain such that the functions decay to $10^{-8}$.

**Fast**
$$
\begin{align}
u_1(x,y)=e^{-(2x^2+3y^2)}& \leq\ 10^{-8}\\
-2x^2+3y^2& \leq -8\times ln10 \\
2x^2+3y^2& \ge 2x^2 \ge 8\times ln(10) \\
x & \ge 3.034854
\end{align}
$$
so I choose the domain $[-5,5]\times[-5,5]$ for the outer NN.

**Slow**
$$
\begin{align}
u_2(x,y)=\frac{1}{x^2+y^2+1}& \leq 10^{-8}\\
\frac{1}{x^2+y^2+1}& \leq\frac{1}{x^2}\leq10^{-8}\\
x^2& \ge10^4\\
x& \ge 100
\end{align}
$$
so I choose the domain $[-100, 100]\times[-100, 100]$ for the outer NN.

## Model 
Both function use the same constructure of DNN model.
The only differece is the collocation points of outer nueral network.

## Compare the performance
I use the method, norm, to compare the approximation performance on 2 different funtion which dacays to 0 with different rate.
And they are defined as below:
$$
\begin{align}
& \|u_{approx}(x,y)-u_{exact}(x,y)\|_1=\int |u_{approx}(x,y)-u_{exact}(x,y)|\ dx\\
& \|u_{approx}(x,y)-u_{exact}(x,y)\|_2=\sqrt{\int |u_{approx}(x,y)-u_{exact}(x,y)|^2\ dx}\\
& \|u_{approx}(x,y)-u_{exact}(x,y)\|_{\infty}=max\{|u_{approx}(x,y)-u_{exact}(x,y)|\}
\end{align}
$$

We are now checking the loss curve over epochs first.
<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/r1XKd5wyZe.png" alt="Figure1" width="80%">
</div>

<p class="text-center">Fig 1. The loss curve of solution with fast decay over epochs.</p>

<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/HJssDSm1Wg.png" alt="Figure1" width="80%">
</div>

<p class="text-center">Fig 2. The loss curve of solution with slow decay over epochs.</p>


Following, the predicted solution and its error are dipicted in the following.
<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/Bkxhu5D1be.png" alt="Figure1" width="100%">
</div>

<p class="text-center">Fig 3. The predicted solution of PINNs and the absolute error between true solution and predicted solution with fast decay solution.</p>

<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/rJtEFqP1Ze.png" alt="Figure1" width="100%">
</div>

<p class="text-center">Fig 3. The predicted solution of PINNs and the absolute error between true solution and predicted solution with slow decay solution.</p>

Then, the table show different norms of two target solutions.

|  | $e^{-(2x^2+3y^2)}$ | $\frac{1}{x^2+y^2+1}$ |
| -------- | -------- | -------- |
| $L_1$ | $5.76867$ | $2.55511$ |
| $L_2$ | $3.25477$ | $1.46271$ |
| $L_{\infty}$ | $1.88189$ | $1.12017$ |