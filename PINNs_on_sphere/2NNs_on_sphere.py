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

# True solution on sphere
@tf.function
def tru(x, y, z):
    
    return tru
# The associated x derivative
@tf.function
def trudx(x, y, z):
    
    return trudx

# The associated y derivative
@tf.function
def trudy(x, y, z):
    
    return trudy

@tf.function
def trudz(x, y, z):
    
    return trudz

# =============================
# Data Generation for NN_1
# =============================
# Boundary points for NN_1


# Collocation points for NN_1


# =============================
# Data Generation for NN_2
# =============================


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

@tf.function
def u1(x, y, z):
    u1 = model_1(tf.concat([x, y, z], axis=1))
    return u1

# Residual equation of NN_1(PDE_Loss of NN_1)
@tf.function
def f1(x, y, z):
    u1_0 = u1(x, y, z)
    u1_x = tf.GradientTape().gradient(u1_0, x)[0]
    u1_xx = tf.GradientTape().gradient(u1_x, x)[0]
    u1_y = tf.GradientTape().gradient(u1_0, y)[0]
    u1_yy = tf.GradientTape().gradient(u1_y, y)[0]
    u1_z = tf.GradientTape().gradient(u1_0, z)[0]
    u1_zz = tf.GradientTape().gradient(u1_z, z)[0]

    F1 = u1_xx + u1_yy + u1_zz - (2*(8*x**4+2*x**2*(12*y**2+(1-z)**2)+18*y**4+9*y**2*(1-z)**2+5*(1-z)**4)/(1-z)**8)*tf.exp(-((2*x**2) + (3*y**2)/(1-z)**2))
    retour = tf.reduce_mean(tf.square(F1))
    return retour

@tf.function
def u2(x, y, z):
    u2 = model_2(tf.concat([x, y, z], axis=1))
    return u2

# Residual equation of NN_2(PDE_Loss of NN_2)
@tf.function
def f2(x, y, z):
    u2_0 = u2(x, y, z)
    u2_x = tf.GradientTape().gradient(u2_0, x)[0]
    u2_xx = tf.GradientTape().gradient(u2_x, x)[0]
    u2_y = tf.GradientTape().gradient(u2_0, y)[0]
    u2_yy = tf.GradientTape().gradient(u2_y, y)[0]
    u2_z = tf.GradientTape().gradient(u2_0, z)[0]
    u2_zz = tf.GradientTape().gradient(u2_z, z)[0]

    F2 = u2_xx + u2_yy + u2_zz -  (2*(8*x**4+2*x**2*(12*y**2+(1-z)**2)+18*y**4+9*y**2*(1-z)**2+5*(1-z)**4)/(1-z)**8)*tf.exp(-((2*x**2) + (3*y**2)/(1-z)**2))
    retour = tf.reduce_mean(tf.square(F2))
    return retour

# Data_Loss of the neural networks
@tf.function
def mse(u1, u2):
    return tf.reduce_mean(tf.square(u1, u2))

# ============================
# Training begin
# ============================
loss = 0
epochs = 10000
opt1 = tf.keras.optimizers.Adam(learning_rate=1e-4)
opt2 = tf.keras.optimizers.Adam(learning_rate=1e-4)
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
        T1_ = u1(x_b, y_b, z_b)
        T2_ = u2(x_b, y_b, z_b)

        # PDE loss of NN_1
        L1 = 1 * f1(x_south, y_south,z_south)
        # PDE loss of NN_2
        L2 = 1 * f2(x_north, y_north,z_north)
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