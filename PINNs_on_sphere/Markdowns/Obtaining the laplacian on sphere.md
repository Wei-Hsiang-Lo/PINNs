---
title: Obtaining the laplacian on ball
tags: [PINNs]

---

# <p text="text-center"> Obtain the Laplacian on Sphere</p>

Here we want to tranform the PDE $g_{uu}+g_{vv}=f(u,v)$ to a sphere with solution $h(r,\theta,\phi)$ to satisfy it.

## Find the relation between u, v and r, $\theta$, $\phi$
Consider a unit ball, we have:
$$
\begin{align}
& x=rsin(\theta)cos(\phi) \\
& y=rsin(\theta)sin(\phi) \\
& z=rcos(\theta)
\end{align}
$$
where $r=1$, $\theta\in[0,\pi]$ and $\phi\in[0,2\pi]$.

With the former result, we have the following relationship:
$$
x=\frac{2u}{(1+u^2+v^2)^2}\qquad,\qquad y=\frac{2v}{(1+u^2+v^2)^2}\qquad,\qquad z=\frac{u^2+v^2-1}{u^2+v^2+1}
$$
By replacing $x$, $y$, $z$, we will have:
$$
rsin(\theta)cos(\phi)=\frac{2u}{1+u^2+v^2}\quad,\quad rsin(\theta)sin(\phi)=\frac{2v}{1+u^2+v^2}\quad,\quad
rcos(\theta)=\frac{u^2+v^2-1}{1+u^2+v^2}
$$
By the relation above we can have:
$$
\begin{align}
cos\theta& =z=\frac{u^2+v^2-1}{1+u^2+v^2}\\
\theta& =arccos(\frac{u^2+v^2-1}{1+u^2+v^2})\\
\theta& =2arctan(\sqrt{u^2+v^2})
\end{align}
$$
and, 
$$
\begin{align}
tan\phi& = \frac{y}{x}=\frac{\frac{2v}{u^2+v^2+1}}{\frac{2u}{u^2+v^2+1}}=\frac{v}{u}\\
\phi& =acrtan(\frac{v}{u})
\end{align}
$$

With these relationship, we want to derive the partial derivative of $r$, $\theta$ $\phi$ on $u$ and $v$, respectively.

## Derive the partial derivative
We want to represent
$\frac{\partial\theta}{\partial u}$, $\frac{\partial\theta}{\partial v}$, $\frac{\partial\phi}{\partial u}$, $\frac{\partial\phi}{\partial v}$ with $\theta$ and $\phi$.
We apply the inverse function theorem:
$$
[Df^{-1}(y)]=[Df(f-1(y))]^{-1}
$$
Hence,
$$
\begin{bmatrix}
\frac{\partial\theta}{\partial u} & \frac{\partial\theta}{\partial v}\\
\frac{\partial\phi}{\partial u} & \frac{\partial\phi}{\partial v}\end{bmatrix}=
\begin{bmatrix}
\frac{\partial u}{\partial\theta} & \frac{\partial u}{\partial\phi}\\
\frac{\partial v}{\partial\theta} & \frac{\partial v}{\partial\phi}
\end{bmatrix}^{-1}=\frac{1}{|J|}
\begin{bmatrix}
\frac{\partial v}{\partial\phi} & -\frac{\partial u}{\partial\phi}\\
-\frac{\partial v}{\partial\theta} & \frac{\partial u}{\partial\theta}
\end{bmatrix}
$$

By the former result, we have $u=\frac{sin\theta cos\phi}{1-cos\theta}, v=\frac{sin\theta sin\phi}{1-cos\theta}$
So we can derive $\frac{\partial u}{\partial\theta}, \frac{\partial u}{\partial\phi}, \frac{\partial v}{\partial\theta}, \frac{\partial v}{\partial\phi}$.
$$
\begin{align}
\frac{\partial u}{\partial\theta}& =\frac{(1-cos\theta)cos\theta cos\phi-sin^2\theta cos\phi}{(1-cos\theta)^2}\\
& =\frac{cos\phi(cos\theta-cos^2\theta-sin^2\theta)}{(1-cos)^2}\\
& =\frac{cos\phi(cos\theta-1)}{(1-cos\theta)^2}=\frac{cos\phi}{1-cos\theta}\\
\frac{\partial u}{\partial\phi}&=\frac{-sin\theta sin\phi}{1-cos\theta}\\
\frac{\partial v}{\partial\theta}& =\frac{(1-cos\theta)cos\theta cos\phi-sin^2\theta sin\phi}{(1-cos\theta)^2}\\
\frac{\partial v}{\partial\phi}& =\frac{sin\theta cos\phi}{1-cos\theta}
\end{align}
$$
Then, we calculate $|J|$,
$$
|J|=\frac{\partial u}{\partial\theta}\frac{\partial v}{\partial\phi}-\frac{\partial u}{\partial\phi}\frac{\partial v}{\partial\theta}=\frac{sin\theta(-cos^2\phi-sin^2\phi)}{(1-cos\theta)^2}=\frac{-sin\theta}{(1-cos\theta)^2}
$$

As a result, we have:
$$
\begin{bmatrix}
\frac{\partial\theta}{\partial u} & \frac{\partial\theta}{\partial v}\\
\frac{\partial\phi}{\partial u} & \frac{\partial\phi}{\partial v}\end{bmatrix}=
\begin{bmatrix}
-(1-cos\theta)cos\phi & -(1-cos\theta)sin\phi \\
\frac{-1+cos\theta}{sin\theta}sin\phi & \frac{1-cos\theta}{sin\theta}cos\phi
\end{bmatrix}
$$

## Derive the Laplacian
Assume the true solution on $\mathbb{R}^2$ and on sphere are $g(u,v)$ and $h(1,\theta(u,v),\phi(u,v)$, respectively, with $g(u,v)=h(1, \theta,\phi)$
$$
\begin{bmatrix}
g_u& g_v
\end{bmatrix}=
\begin{bmatrix}
h_{\theta}& h_{\phi}
\end{bmatrix}
\begin{bmatrix}
-(1-cos\theta)cos\phi & -(1-cos\theta)sin\phi \\
\frac{-1+cos\theta}{sin\theta}sin\phi & \frac{1-cos\theta}{sin\theta}cos\phi
\end{bmatrix}
$$
Following,
$$
\begin{bmatrix}
g_{uu}& g_{uv}\\
g_{vu}& g_{vv}
\end{bmatrix}=
\begin{bmatrix}
g_{u\theta}& g_{v\phi} \\
g_{u\theta}& g_{v\phi}
\end{bmatrix}
\begin{bmatrix}
-(1-cos\theta)cos\phi & -(1-cos\theta)sin\phi \\
\frac{-1+cos\theta}{sin\theta}sin\phi & \frac{1-cos\theta}{sin\theta}cos\phi
\end{bmatrix}
$$