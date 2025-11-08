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
def tru(theta, phi):
    # Supposed that the true solution on R^2 is tf.exp(-(2*x**2+3*y**2))
    tru=tf.exp(-(2*(tf.sin(theta)*tf.cos(phi)/(1-tf.cos(theta)))**2+3*(tf.sin(theta)*tf.sin(phi)/(1-tf.cos(theta)))**2))
    return tru

# The associated x derivative
@tf.function
def trudt(theta, phi):
    trudt=tf.exp(-(2*(tf.sin(theta)*tf.cos(phi)/(1-tf.cos(theta)))**2+3*(tf.sin(theta)*tf.sin(phi)/(1-tf.cos(theta)))**2))*((2*tf.sin(theta)*(tf.sin(theta)**2+tf.cos(theta)**2-tf.cos(theta))*(3*tf.sin(phi)**2+2*tf.cos(phi)**2))/((1-tf.cos(theta))**3))
    return trudt

# The associated y derivative
@tf.function
def trudp(theta, phi):
    trudp=tf.exp(-(2*(tf.sin(theta)*tf.cos(phi)/(1-tf.cos(theta)))**2+3*(tf.sin(theta)*tf.sin(phi)/(1-tf.cos(theta)))**2))*(-(2*tf.sin(theta)**2*tf.sin(phi)*tf.cos(phi))/((1-tf.cos(theta))**2))
    return trudp

@tf.function
def trudtt(theta, phi):
    eps = tf.constant(1e-12, dtype=tf.float64)  # 避免除以零
    denom = tf.maximum(1 - tf.cos(theta), eps)

    sin_theta = tf.sin(theta)
    cos_theta = tf.cos(theta)
    sin_phi = tf.sin(phi)
    cos_phi = tf.cos(phi)

    # u, v
    u = sin_theta * cos_phi / denom
    v = sin_theta * sin_phi / denom

    # u_theta, v_theta
    u_t = (cos_theta*denom - sin_theta**2) * cos_phi / denom**2
    v_t = (cos_theta*denom - sin_theta**2) * sin_phi / denom**2

    # u_tt, v_tt
    u_tt = (-(sin_theta*denom + 2*u_t*denom - 2*sin_theta*cos_theta) * cos_phi) / denom**2
    v_tt = (-(sin_theta*denom + 2*v_t*denom - 2*sin_theta*cos_theta) * sin_phi) / denom**2

    # E, F
    E = tf.exp(-(2*u**2 + 3*v**2))
    F = 2 * sin_theta * (3*sin_phi**2 + 2*cos_phi**2) / denom**2

    # E_theta, E_tt
    E_t = -E * (4*u*u_t + 6*v*v_t)
    E_tt = -E_t*(4*u*u_t + 6*v*v_t) - E*(4*(u_t**2 + u*u_tt) + 6*(v_t**2 + v*v_tt))

    # F_theta, F_tt
    F_t = 2*(3*sin_phi**2 + 2*cos_phi**2)*(cos_theta*denom - 2*sin_theta**2)/denom**3
    # 完整展開 F_tt（用商法則）：
    F_tt = 2*(3*sin_phi**2 + 2*cos_phi**2) * (
        (-sin_theta*denom**3 - 6*sin_theta*cos_theta*denom**2 + 6*sin_theta**2*denom) / denom**6
    )

    # 二階偏導
    return E_tt*F + 2*E_t*F_t + E*F_tt

@tf.function
def trudtp(theta, phi):
    eps = tf.constant(1e-12, dtype=tf.float64)  # 防止除以零
    denom = tf.maximum(1 - tf.cos(theta), eps)

    sin_theta = tf.sin(theta)
    cos_theta = tf.cos(theta)
    sin_phi = tf.sin(phi)
    cos_phi = tf.cos(phi)

    # u, v
    u = sin_theta * cos_phi / denom
    v = sin_theta * sin_phi / denom
    
    # u_phi, v_phi
    u_phi = - sin_theta * sin_phi / denom
    v_phi = sin_theta * cos_phi / denom
    
    # E, F
    E = tf.exp(-(2*u**2 + 3*v**2))
    F = 2 * sin_theta * (3*sin_phi**2 + 2*cos_phi**2) / denom**2
    
    # F_phi
    F_phi = 4 * sin_theta * sin_phi * cos_phi / denom**2
    
    return - E * (4*u*u_phi + 6*v*v_phi) * F + E * F_phi

@tf.function
def trudpt(theta, phi):
    eps = tf.constant(1e-12, dtype=tf.float64)  # 避免除以0
    denom = tf.maximum(1 - tf.cos(theta), eps)

    sin_theta = tf.sin(theta)
    cos_theta = tf.cos(theta)
    sin_phi = tf.sin(phi)
    cos_phi = tf.cos(phi)

    # u, v
    u = sin_theta * cos_phi / denom
    v = sin_theta * sin_phi / denom

    # u_theta, v_theta
    u_theta = (cos_theta * denom - sin_theta**2) * cos_phi / denom**2
    v_theta = (cos_theta * denom - sin_theta**2) * sin_phi / denom**2

    # E, G
    E = tf.exp(-(2*u**2 + 3*v**2))
    G = -2 * sin_theta**2 * sin_phi * cos_phi / denom**2

    # G_theta
    G_theta = -4 * sin_theta * (cos_theta * denom - sin_theta**2) * sin_phi * cos_phi / denom**3

    return - E * (4*u*u_theta + 6*v*v_theta) * G + E * G_theta

@tf.function
def trudpp(theta, phi):
    eps = tf.constant(1e-12, dtype=tf.float64)  # 避免除以零
    denom = tf.maximum(1 - tf.cos(theta), eps)

    sin_theta = tf.sin(theta)
    cos_theta = tf.cos(theta)
    sin_phi = tf.sin(phi)
    cos_phi = tf.cos(phi)
    
    # u, v
    u = sin_theta * cos_phi / denom
    v = sin_theta * sin_phi / denom
    
    # u_phi, v_phi
    u_phi = - sin_theta * sin_phi / denom
    v_phi = sin_theta * cos_phi / denom
    
    # E, G
    E = tf.exp(-(2*u**2 + 3*v**2))
    G = -2 * sin_theta**2 * sin_phi * cos_phi / denom**2
    
    # G_phi
    G_phi = -2 * sin_theta**2 * (cos_phi**2 - sin_phi**2) / denom**2  # = -2*sin^2θ*cos2φ/(1-cosθ)^2
    
    return - E * (4*u*u_phi + 6*v*v_phi) * G + E * G_phi

# =============================
# Data Generation for NN_1
# =============================
# Boundary points for NN_1(on the equator)
Neq = 1000 
engine = qmc.LatinHypercube(d=1)
theta1 = np.full(Neq, np.pi/2)
phi1 = 2 * np.pi * engine.random(Neq)[:, 0]
data = np.zeros((Neq, 2))
data[:, 0] = theta1
data[:, 1] = phi1

#
theta_eq, phi_eq = map(lambda x: np.expand_dims(x, axis=1), [data[:, 0], data[:, 1]])

# Collocation points for NN_1
Nsouth = 5000
engine = qmc.LatinHypercube(d=2)
colloc_south = engine.random(Nsouth*2)

theta2 = np.pi/2 + (np.pi/2) * colloc_south[:, 0]
phi2 = 2 * np.pi * colloc_south[:, 1]

# Ensure points are not on the equator
mask_south = (theta2 > np.pi/2)
theta2_masked = theta2[mask_south][:Nsouth]
phi2_masked = phi2[mask_south][:Nsouth]
#
theta_south, phi_south = map(lambda x: np.expand_dims(x, axis=1), [theta2_masked, phi2_masked])

# Show the 3D scatter plot
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D Scatter Plot of Points on south hemisphere and equator')
ax.scatter(np.sin(data[:, 0]) * np.cos(data[:, 1]), np.sin(data[:, 0]) * np.sin(data[:, 1]), np.cos(data[:, 0]), c='k', marker='x', s=2, label="Boundary Points")
ax.scatter(np.sin(theta_south)*np.cos(phi_south), np.sin(theta_south)*np.sin(phi_south), np.cos(theta_south), c='b', marker='o', s=2, label='Collocation Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# =============================
# Data Generation for NN_2
# =============================
# Collocation points for NN_2 (North hemisphere)
Nnorth = 50000
engine = qmc.LatinHypercube(d=2)
colloc_north = engine.random(Nnorth*2)

theta3 = (np.pi/2) * colloc_north[:, 0]
phi3 = 2 * np.pi * colloc_north[:, 1]

# Ensure points are not on the equator and north pole
mask_north = (theta3 < np.pi/2) & (theta3 > 0) 
theta3_masked = theta3[mask_north][:Nnorth]
phi3_masked = phi3[mask_north][:Nnorth]

#
theta_north, phi_north = map(lambda x: np.expand_dims(x, axis=1), [theta3_masked, phi3_masked])

# Show the 3D scatter plot
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D Scatter Plot of Points on north hemisphere')
ax.scatter(np.sin(theta_north)*np.cos(phi_north), np.sin(theta_north)*np.sin(phi_north), np.cos(theta_north), c='r', marker='o', s=2, label='Collocation Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')  
ax.set_zlabel('Z')
plt.show()

# Convert np arrays to tf tensors
theta_eq, phi_eq, theta_south, phi_south, theta_north, phi_north = map(lambda x: tf.convert_to_tensor(x, dtype=tf.float64), [theta_eq, phi_eq, theta_south, phi_south, theta_north, phi_north])

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
tf.keras.utils.plot_model(model_1, show_shapes=True, show_layer_names=True, show_dtype=True, show_layer_activations=True)

tf.keras.backend.clear_session()
model_2 = DNN_builder(f"DNN-2{6}", 2, 1, 6, 32, "tanh")
model_2.summary()
tf.keras.utils.plot_model(model_2, show_shapes=True, show_layer_names=True, show_dtype=True, show_layer_activations=True)

@tf.function
def u1(theta, phi):
    u1=model_1(tf.concat([theta, phi], axis=1))
    return u1

# Residual equation of NN_1(PDE_Loss of NN_1)
@tf.function
def f1(theta, phi):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([theta, phi])
        h_0=u1(theta, phi)
        # Write down the Jacobian matrix
        t_u=(-(1-tf.cos(theta))*tf.cos(phi))
        p_u=(-(1-tf.cos(theta))/tf.sin(theta)*tf.sin(phi))
        t_v=(-(1-tf.cos(theta))*tf.sin(phi))
        p_v=((1-tf.cos(theta))/tf.sin(theta)*tf.cos(phi))

    
        h_t=tape.gradient(h_0, theta)
        h_p=tape.gradient(h_0, phi)
        # The Jacobian matrix multiplication
        g_u=h_t*t_u + h_p*p_u
        g_v=h_t*t_v + h_p*p_v

        # The second partial derivatives
        g_ut=tape.gradient(g_u, theta)
        g_up=tape.gradient(g_u, phi)
        g_vt=tape.gradient(g_v, theta)
        g_vp=tape.gradient(g_v, phi)
        # The Jacobian matrix multiplication
        g_uu=g_ut*t_u + g_up*p_u
        g_vv=g_vt*t_v + g_vp*p_v

        # Calculate the residual equation
        tru_ut=-trudtt(theta, phi)*(1-tf.cos(theta))*tf.cos(phi) - trudt(theta, phi)*tf.sin(theta)*tf.cos(phi) - trudpt(theta, phi)*((1-tf.cos(theta))/tf.sin(theta)*tf.sin(phi)) - trudp(theta, phi)*((tf.sin(theta)**2-tf.cos(theta)**2+tf.cos(theta))/(tf.sin(theta)**2)*tf.sin(phi))
        tru_vt=-trudtt(theta, phi)*(1-tf.cos(theta))*tf.sin(phi) - trudt(theta, phi)*tf.sin(theta)*tf.sin(phi) - trudpt(theta, phi)*((1-tf.cos(theta))/tf.sin(theta)*tf.cos(phi)) - trudp(theta, phi)*((tf.sin(theta)**2-tf.cos(theta)**2+tf.cos(theta))/(tf.sin(theta)**2)*tf.cos(phi))
        tru_up=-trudtp(theta, phi)*(1-tf.cos(theta))*tf.cos(phi) + trudt(theta, phi)*(1-tf.cos(theta))*tf.sin(phi) - trudpp(theta, phi)*(1-tf.cos(theta))/tf.sin(theta)*tf.sin(phi) - trudp(theta, phi)*((1-tf.cos(theta))/tf.sin(theta))*tf.cos(phi)
        tru_vp=-trudtp(theta, phi)*(1-tf.cos(theta))*tf.sin(phi) - trudt(theta, phi)*(1-tf.cos(theta))*tf.cos(phi) - trudpp(theta, phi)*(1-tf.cos(theta))/tf.sin(theta)*tf.cos(phi) + trudp(theta, phi)*((1-tf.cos(theta))/tf.sin(theta))*tf.sin(phi)

        F1=g_uu + g_vv - (tru_ut*t_u+tru_up*p_u + tru_vt*t_v + tru_vp*p_v)
        retour = tf.reduce_mean(tf.square(F1))
    return retour

@tf.function
def u2(theta, phi):
    u2=model_2(tf.concat([theta, phi], axis=1))
    return u2

# Residual equation of NN_2(PDE_Loss of NN_2)
@tf.function
def f2(theta, phi):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([theta, phi])
        h_0=u2(theta, phi)
        # Write down the Jacobian matrix
        t_u=(-(1-tf.cos(theta))*tf.cos(phi))
        p_u=(-(1-tf.cos(theta))/tf.sin(theta)*tf.sin(phi))
        t_v=(-(1-tf.cos(theta))*tf.sin(phi))
        p_v=((1-tf.cos(theta))/tf.sin(theta)*tf.cos(phi))

        h_t=tape.gradient(h_0, theta)
        h_p=tape.gradient(h_0, phi)
        # The Jacobian matrix multiplication
        g_u=h_t*t_u + h_p*p_u
        g_v=h_t*t_v + h_p*p_v

        # The second partial derivatives
        g_ut=tape.gradient(g_u, theta)
        g_up=tape.gradient(g_u, phi)
        g_vt=tape.gradient(g_v, theta)
        g_vp=tape.gradient(g_v, phi)
        # The Jacobian matrix multiplication
        g_uu=g_ut*t_u + g_up*p_u
        g_vv=g_vt*t_v + g_vp*p_v

        # Calculate the residual equation
        tru_ut=-trudtt(theta, phi)*(1-tf.cos(theta))*tf.cos(phi) - trudt(theta, phi)*tf.sin(theta)*tf.cos(phi) - trudpt(theta, phi)*((1-tf.cos(theta))/tf.sin(theta)*tf.sin(phi)) - trudp(theta, phi)*((tf.sin(theta)**2-tf.cos(theta)**2+tf.cos(theta))/(tf.sin(theta)**2)*tf.sin(phi))
        tru_vt=-trudtt(theta, phi)*(1-tf.cos(theta))*tf.sin(phi) - trudt(theta, phi)*tf.sin(theta)*tf.sin(phi) - trudpt(theta, phi)*((1-tf.cos(theta))/tf.sin(theta)*tf.cos(phi)) - trudp(theta, phi)*((tf.sin(theta)**2-tf.cos(theta)**2+tf.cos(theta))/(tf.sin(theta)**2)*tf.cos(phi))
        tru_up=-trudtp(theta, phi)*(1-tf.cos(theta))*tf.cos(phi) + trudt(theta, phi)*(1-tf.cos(theta))*tf.sin(phi) - trudpp(theta, phi)*(1-tf.cos(theta))/tf.sin(theta)*tf.sin(phi) - trudp(theta, phi)*((1-tf.cos(theta))/tf.sin(theta))*tf.cos(phi)
        tru_vp=-trudtp(theta, phi)*(1-tf.cos(theta))*tf.sin(phi) - trudt(theta, phi)*(1-tf.cos(theta))*tf.cos(phi) - trudpp(theta, phi)*(1-tf.cos(theta))/tf.sin(theta)*tf.cos(phi) + trudp(theta, phi)*((1-tf.cos(theta))/tf.sin(theta))*tf.sin(phi)

        F2=g_uu + g_vv - (tru_ut*t_u+tru_up*p_u + tru_vt*t_v + tru_vp*p_v)
        retour = tf.reduce_mean(tf.square(F2))
    return retour

# Data_Loss of the neural networks
@tf.function
def mse(u1, u2):
    return tf.reduce_mean(tf.square(u1 - u2))

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
        T1_ = u1(theta_eq, phi_eq)
        T2_ = u2(theta_eq, phi_eq)

        # PDE loss of NN_1
        L1 = 1 * f1(theta_south, phi_south)
        # PDE loss of NN_2
        L2 = 1 * f2(theta_north, phi_north)
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

# ============================
# Use Simpson's Rule to calculate the error
# ============================
# Derive grid s.t. the accuracy to O(10^{-8})
Nr = 151
Ntheta = 101

r = np.linspace(0, 1, Nr)
theta = np.linspace(0, 2*np.pi, Ntheta)
R_, Theta_ = np.meshgrid(r, theta, indexing='ij')
X = R_*np.cos(Theta_)
Y = R_*np.sin(Theta_)

# Convert numpy data to tensor
x_tf = tf.convert_to_tensor(X.flatten().reshape(-1, 1), dtype=tf.float64)
y_tf = tf.convert_to_tensor(Y.flatten().reshape(-1, 1), dtype=tf.float64)

U_pred = u1(tf.atan(tf.sqrt(tf.square(x_tf)+tf.square(y_tf))), tf.atan2(y_tf/x_tf)).numpy().reshape(Nr, Ntheta)
U_true = tru(tf.atan(tf.sqrt(tf.square(x_tf)+tf.square(y_tf))), tf.atan2(y_tf/x_tf)).numpy().reshape(Nr, Ntheta)
E = np.abs(U_pred-U_true)

# === Obtain the L-1, L-2, L-inf norms ===
L1_norm = integrate.simpson(integrate.simpson(E * R_, theta), r)
L2_norm = np.sqrt(integrate.simpson(integrate.simpson(E**2 * R_, theta), r))
Linf_norm = np.max(E)

print(f"L1 norm: {L1_norm:.5e}")
print(f"L2 norm: {L2_norm:.5e}")
print(f"Linf norm: {Linf_norm:.5e}")