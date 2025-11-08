---
title: Stereographic Projection
tags: [PINNs]

---

# <p class="text-center">Stereographic Projection</p>
## Project 2D data point on the surface of a unit ball
### Idea
Let $(u,v)\in\mathbb{R}^2, S=\{(x,y,z)\ |\ x^2+y^2+z^2=1\}$

<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/rysOEbrCxl.png" alt="Figure1" width="80%">
</div>

In this figure, we want to project $z$ onto the shpere of a unit ball. 
We incident $Z$ with the north pole $N=(0,0,1)$ and the intersection of sphere and $NZ$-line is the desired projection point of $Z$, $P=(x,y,z)$.

Since we want to embed the neural networks(on $\mathbb{R}^2$) into the sphere.
We need to represent $(u,v,0)$ with $(x,y,z)$.

### Transform the Coordinate
<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/SkucaZHCgg.png" alt="Figure1" width="80%">
</div>

**Notation**
* $N$: north pole of the unit ball,
* $P$: the projection point of $Z$ on the shpere,
* $P^{\prime}$: the projection of P onto the vector $\overrightarrow{Z}$

From this figure, we have:
$$
(u,v,0)=\frac{\sqrt{u^2+v^2}}{\sqrt{x^2+y^2}}(x,y,0)
$$

<div style="text-align: center;">
    <img src="https://hackmd.io/_uploads/Hyx3ezHAlg.png" alt="Figure1" width="80%">
</div>

**Notation**
* $|Z\ |$: The distance between $(0,0,0)$ and $Z$,
* $|P^\prime|$: The distance between $(0,0,0)$ and $P^\prime$.

Since it is a right triangle, we have:
$$
tan(\theta)=\frac{z}{|Z|-|P^\prime|}=\frac{1}{|Z|}
$$
By this, we obtain $z=1-\frac{|P^\prime|}{|Z\ |}$, $\frac{|Z|}{|P^\prime|}=\frac{1}{1-z}$.
Since $|P^\prime|=\sqrt{x^2+y^2}$ and $|Z|=\sqrt{u^2+v^2}$, we have the folling relationship:
$$
(u,v,0)=\frac{1}{1-z}(x,y,0)\Rightarrow u=\frac{x}{1-z}, v=\frac{y}{1-z}
$$

