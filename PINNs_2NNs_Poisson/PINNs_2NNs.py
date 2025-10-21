import numpy as np
import time 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import qmc
from scipy import integrate

# Set float precision to double
tf.keras.backend.set_floatx("float64")

# .......................................
# We solve Poissons equation Delta u = f
#  for different f (x,y)
#.........................................

coef = 1.

# Select true solution .........................
@tf.function
def tru(x, y):
    tru=tf.exp(-(2*(x**2)+3*(y**2)))
    #tru=tf.exp(x*y)
    #tru=tf.exp(x)*tf.sin(y)+1/4*(x*x+y*y)
    #tru=tf.sinh(x)
    #tru=tf.exp(x*x+y*y)
    #tru=tf.exp(x*y)+coef*tf.sinh(x)
    return tru

# Select associated x derivatives .........
@tf.function
def trudx(x, y):
    trudx=tf.exp(-(2*(x**2)+3*(y**2)))*(-4*x)
    #trudx=tf.exp(x*y)*y    
    #trudx=tf.exp(x)*tf.sin(y)+1/2*x
    #trudx=tf.cosh(x)    
    #trudx=tf.exp(x*x+y*y)*2*x    
    #trudx=y*tf.exp(x*y)+coef*tf.cosh(x) 
    return trudx

# Select associated y derivatives .........
@tf.function
def trudy(x, y):
    trudy=tf.exp(-(2*(x**2)+3*(y**2)))*(-6*y)
    #trudy=tf.exp(x*y)*x    
    #trudy=tf.exp(x)*tf.cos(y)+1/2*y    
    #trudy=0    
    #trudy=tf.exp(x*x+y*y)*2*y  
    #trudy=x*tf.exp(x*y)
    return trudy

# ============================
# Data generation for NN_1
# ============================
# Boundary points for NN_1
Nb = 200
rmax = 1.0
engine = qmc.LatinHypercube(d=1)
theta = 2 * np.pi * engine.random(Nb)[:, 0]
data = np.zeros((Nb, 5))
data[:, 0] = rmax * np.cos(theta)
data[:, 1] = rmax * np.sin(theta)

x_b, y_b = map(lambda x: np.expand_dims(x, axis=1),[data[:, 0], data[:, 1]])

plt.scatter(data[:, 0], data[:, 1], c='k', marker='x', label='Boundary points')

# Collocation points for NN_1
Nc = 1000
engine = qmc.LatinHypercube(d=2)
colloc = engine.random(n=Nc)

r = rmax * np.sqrt(colloc[:, 0])
theta = 2 * np.pi * colloc[:, 1]
# Transform the points to Cartesian coordinates
colloc[:, 0] = r * np.cos(theta)
colloc[:, 1] = r * np.sin(theta)

#
x_c, y_c = map(lambda x: np.expand_dims(x, axis=1), [colloc[:, 0], colloc[:, 1]])
#
plt.figure("", figsize=(7, 7))
plt.title("Boundary points and collocation points for NN_1", fontsize = 16)
plt.scatter(data[:,0], data[:,1], marker="x", c="k", label="BDP")
plt.scatter(colloc[:,0], colloc[:,1], s=2, marker=".", c="r", label="CP")
plt.xlabel("x",fontsize=16)
plt.ylabel("y",fontsize=16)
plt.axis("square")
plt.show()

#
x_c, y_c, x_b, y_b = map(lambda x: tf.convert_to_tensor(x, dtype=tf.float64), [x_c, y_c, x_b, y_b])

# ============================
# data generation for NN_2
# ============================
N2 = 10000
engine = qmc.LatinHypercube(d=2)
points = np.zeros([N2, 2])
points = engine.random(4*N2)

x_ = 10 * points[:, 0] - 5
y_ = 10 * points[:, 1] - 5

mask = x_**2 + y_**2 > rmax**2
x__ = x_[mask][:N2]
y__ = y_[mask][:N2]

x_out = tf.convert_to_tensor(x__.reshape(-1, 1), dtype=tf.float64)
y_out = tf.convert_to_tensor(y__.reshape(-1, 1), dtype=tf.float64)

plt.figure("", figsize=(7, 7))
plt.title("Collocation points for NN_2", fontsize = 16)
plt.scatter(x__, y__, s=2, marker=".", c="r", label="CP")
plt.xlabel("x",fontsize=16)
plt.ylabel("y",fontsize=16)
plt.axis("square")
plt.show()

# =============================
# Model builder
# =============================
def DNN_builder(name, in_shape = 2, out_shape = 1, hidden_layers = 6, neurons = 32, actfn="tanh"):
    # Input layer
    input_layer = tf.keras.layers.Input(shape=(in_shape,))
    # Hidden layers
    hidden = [tf.keras.layers.Dense(neurons, activation = actfn)(input_layer)]
    for i in range(hidden_layers - 1):
        new_layer = tf.keras.layers.Dense(neurons, activation=actfn, activity_regularizer=None)(hidden[-1])
        hidden.append(new_layer)
    # Output layer
    output_layer = tf.keras.layers.Dense(1, activation=None)(hidden[-1])
    # Build model
    model = tf.keras.Model(input_layer, output_layer, name=name)
    return model

tf.keras.backend.clear_session()
model_1 = DNN_builder(f"DNN-1{6}", 2, 1, 6, 32, "tanh")
model_1.summary()
tf.keras.utils.plot_model(model_1, to_file='NN_1_plot.png', show_shapes=True, show_layer_names=True, show_dtype=True, show_layer_activations=True)

tf.keras.backend.clear_session()
model_2 = DNN_builder(f"DNN-2{6}", 2, 1, 6, 32, "tanh")
model_2.summary()
tf.keras.utils.plot_model(model_2, to_file='NN_2_plot.png', show_shapes=True, show_layer_names=True, show_dtype=True, show_layer_activations=True)

# =============================
# Calculate u1 and its partial derivatives
# =============================
@tf.function
def u1(x, y):
    u1 = model_1(tf.concat([x, y], axis=1))
    return u1

@tf.function
def u1derx(x, y):
    u1 = model_1(tf.concat([x, y], axis=1))
    u1derx = tf.gradients(u1, x)[0]
    return u1derx

@tf.function
def u1dery(x, y):
    u1 = model_1(tf.concat([x, y], axis=1))
    u1dery = tf.gradients(u1, y)[0]
    return u1dery

# PDE Loss of u1
@tf.function
def f1(x, y):
    u1_0= u1(x, y)
    u1_x = tf.gradients(u1_0, x)[0]
    u1_xx = tf.gradients(u1_x, x)[0]
    u1_y = tf.gradients(u1_0, y)[0]
    u1_yy = tf.gradients(u1_y, y)[0]

    # Select the redidual equation
    F1 = u1_xx + u1_yy - (16*x*x+36*y*y-10)*tf.exp(-(2*(x**2)+3*(y**2)))
    #F1 = u1_xx + u1_yy - (x*x+y*y)*tf.exp(x*y)
    #F1 = u1_xx + u1_yy - 1
    #F1 = u1_xx + u1_yy - tf.sinh(x)
    #F1 = u1_xx+u1_yy -4*(x*x+y*y+1)*tf.exp(x*x+y*y)
    retour = tf.reduce_mean(tf.square(F1))
    return retour

# =============================
# Calculate u2 and its partial derivatives
# =============================
@tf.function
def u2(x, y):
    u2 = model_2(tf.concat([x, y], axis=1))
    return u2

@tf.function
def u2derx(x, y):
    u2 = model_2(tf.concat([x, y], axis=1))
    u2derx = tf.gradients(u2, x)[0]
    return u2derx

@tf.function
def u2dery(x, y):
    u2 = model_2(tf.concat([x, y], axis=1))
    u2dery = tf.gradients(u2, y)[0]
    return u2dery

@tf.function
def f2(x, y):
    u2_0 = u2(x, y)
    u2_x = tf.gradients(u2_0, x)[0]
    u2_xx = tf.gradients(u2_x, x)[0]
    u2_y = tf.gradients(u2_0, y)[0]
    u2_yy = tf.gradients(u2_y, y)[0]

    # Select the redidual equation
    F2 = u2_xx + u2_yy - (16*x*x+36*y*y-10)*tf.exp(-(2*(x**2)+3*(y**2)))
    #F2 = u2_xx + u2_yy - (x*x+y*y)*tf.exp(x*y)
    #F2 = u2_xx + u2_yy - 1
    #F2 = u2_xx + u2_yy - tf.sinh(x)
    #F2 = u2_xx+u2_yy -4*(x*x+y*y+1)*tf.exp(x*x+y*y)
    #F2 = u2_xx+u2_yy-(x*x+y*y)*tf.exp(x*y)-coef*tf.sinh(x)
    retour = tf.reduce_mean(tf.square(F2))
    return retour

# Data Loss of the networks
@tf.function
def mse(u1, u2):
    return tf.reduce_mean(tf.square(u1 - u2))

# =============================
# Training begin
# =============================
loss = 0
epochs = 10000
opt1 = tf.keras.optimizers.Adam(learning_rate=1e-2)
opt2 = tf.keras.optimizers.Adam(learning_rate=1e-2)
epoch = 0
loss_values = np.array([]) #total
L1_values = np.array([]) #PDE loss of NN_1
L2_values = np.array([]) #PDE loss of NN_2
l_values = np.array([]) #Mse loss between NN_1 and NN_2 at boundary points

# Record the start time of training NN_2
start = time.time()
# First, we need to train NN_2 first to get its approximation of true_u
for epoch in range(epochs):
    with tf.GradientTape(persistent=True) as tape:
        T1_ = u1(x_b, y_b)
        T2_ = u2(x_b, y_b)

        # PDE loss of NN_1
        L1 = 1 * f1(x_c, y_c)
        # PDE loss of NN_2
        L2 = 1 * f2(x_out, y_out)
        # Mse loss between NN_1 and NN_2 at boundary points
        l = mse(T1_, T2_)
        # Total Loss
        loss = L1 + L2 + l

    g1 = tape.gradient(loss, model_1.trainable_weights)
    opt1.apply_gradients(zip(g1, model_1.trainable_weights))
    g2 = tape.gradient(loss, model_2.trainable_weights)
    opt2.apply_gradients(zip(g2, model_2.trainable_weights))

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"{epoch:5}, {loss.numpy():.9f}")
        loss_values = np.append(loss_values, loss)
        L1_values = np.append(L1_values, L1)
        L2_values = np.append(L2_values, L2)
        l_values = np.append(l_values, l)

# Record the end time of training NN_2
end = time.time()
computation_time_NN2 = {}
computation_time_NN2["PINNs"] = end - start
print(f"\ncomputation time of NNs: {end-start:.3f}\n")

# =============================
# Compute norms after training NN_1
# =============================
# Scalar version of NN_1 prediction
def u1_float_scalar(x, y):
    # x, y are float scalars
    pt = tf.convert_to_tensor([[x, y]], dtype=tf.float64)
    val = u1(pt[:, 0:1], pt[:, 1:2]).numpy()[0, 0]
    return val

# Scalar version of NN_2 prediction
def u2_float_scalar(x, y):
    # x, y are float scalars
    pt = tf.convert_to_tensor([[x, y]], dtype=tf.float64)
    val = u2(pt[:, 0:1], pt[:, 1:2]).numpy()[0, 0]
    return val

# Scalar version of true solution
def tru_float_scalar(x, y):
    # x, y are float scalars
    pt = tf.convert_to_tensor([[x, y]], dtype=tf.float64)
    val = tru(pt[:, 0:1], pt[:, 1:2]).numpy()[0, 0]
    return val

Nr = 151
Ntheta = 101

r = np.linspace(0, rmax, Nr)
theta = np.linspace(0, 2 * np.pi, Ntheta)
R_, Theta_ = np.meshgrid(r, theta, indexing='ij')
X = R_ * np.cos(Theta_)
Y = R_ * np.sin(Theta_)

# Predict multiple data at one time
x_tf = tf.convert_to_tensor(X.flatten().reshape(-1, 1), dtype=tf.float64)
y_tf = tf.convert_to_tensor(Y.flatten().reshape(-1, 1), dtype=tf.float64)

U_pred = u1(x_tf, y_tf).numpy().reshape(Nr, Ntheta)
U_true = tru(x_tf, y_tf).numpy().reshape(Nr, Ntheta)
E = np.abs(U_pred - U_true)

# === Simpson's rule ===
L1_norm = integrate.simpson(integrate.simpson(E * R_, theta), r)
L2_norm = np.sqrt(integrate.simpson(integrate.simpson(E**2 * R_, theta), r))
Linf_norm = np.max(E)

print(f"L1 norm: {L1_norm:.5e}")
print(f"L2 norm: {L2_norm:.5e}")
print(f"Linf norm: {Linf_norm:.5e}")

n = 1000

x = np.linspace(-rmax, rmax, n)
y = np.linspace(-rmax, rmax, n)
X0, Y0 = np.meshgrid(x, y)

# Mask for points inside the circle
mask = X0**2 + Y0**2 <= rmax**2

# Flassten and convert to tensors
X = X0.reshape(-1, 1)
Y = Y0.reshape(-1, 1)
X_T = tf.convert_to_tensor(X)
Y_T = tf.convert_to_tensor(Y)

# Predicted solution by the network
S = u1(X_T, Y_T)
S = S.numpy().reshape(n, n)

# Apply mask (set values ouside circle = NaN for plotting)
S_masked = np.where(mask, S, np.nan)

# Ground truth
TT = tru(X0, Y0)
TT2 = np.where(mask, TT - S, np.nan)

plt.figure("", figsize=(14,7))
# Plot the PINN solution
plt.subplot(221)
plt.pcolormesh(X0, Y0, S_masked, cmap="turbo")
plt.colorbar(pad=-0.3)
plt.xlabel("X", fontsize=16)
plt.ylabel("Y", fontsize=16)
plt.title("PINN solution", fontsize=16)
plt.axis("square")
plt.xlim(-rmax - 0.1, rmax + 0.1)
plt.ylim(-rmax - 0.1, rmax + 0.1)

plt.subplot(222)
plt.pcolormesh(X0, Y0, TT2, cmap="turbo")
plt.colorbar(pad=-0.3)
plt.xlabel("X", fontsize=16)
plt.ylabel("Y", fontsize=16)
plt.title("Absolute error", fontsize=16)
plt.axis("square")
plt.xlim(-rmax - 0.1, rmax + 0.1)
plt.ylim(-rmax - 0.1, rmax + 0.1)

plt.tight_layout()
plt.show()

# Plot loss curves
plt.figure("", figsize=(8,6))
plt.semilogy(loss_values, label="Total Loss")
plt.semilogy(L1_values, label="PDE Loss of NN_1")
plt.semilogy(L2_values, label="PDE Loss of NN_2")
plt.semilogy(l_values, label="Loss_PDE")
plt.xlabel("Epochs"r'($\times 100$)', fontsize=16)
plt.legend()
plt.title("Training Loss Curves", fontsize=16)
plt.grid(True)
plt.show()