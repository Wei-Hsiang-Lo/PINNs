# =============================
# # Model builder
# # =============================
# def DNN_builder(name, in_shape = 2, out_shape = 1, hidden_layers = 6, neurons = 32, actfn="tanh"):
#     # Input layer
#     input_layer = tf.keras.layers.Input(shape=(in_shape,))
#     # Hidden layers
#     hidden = [tf.keras.layers.Dense(neurons, activation = actfn)(input_layer)]
#     for i in range(hidden_layers - 1):
#         new_layer = tf.keras.layers.Dense(neurons, activation=actfn, activity_regularizer=None)(hidden[-1])
#         hidden.append(new_layer)
#     # Output layer
#     output_layer = tf.keras.layers.Dense(1, activation=None)(hidden[-1])
#     # Build model
#     model = tf.keras.Model(input_layer, output_layer, name=name)
#     return model


# tf.keras.backend.clear_session()
# model_1 = DNN_builder(f"DNN-1{6}", 2, 1, 6, 32, "tanh")
# model_1.summary()
# tf.keras.utils.plot_model(model_1, to_file='NN_1_plot.png', show_shapes=True, show_layer_names=True, show_dtype=True, show_layer_activations=True)

# tf.keras.backend.clear_session()
# model_2 = DNN_builder(f"DNN-2{6}", 2, 1, 6, 32, "tanh")
# model_2.summary()
# tf.keras.utils.plot_model(model_2, to_file='NN_2_plot.png', show_shapes=True, show_layer_names=True, show_dtype=True, show_layer_activations=True)

# @tf.function
# def u1(theta, phi):
#     u1=model_1(tf.concat([theta, phi], axis=1))
#     return u1

# # Residual equation of NN_1(PDE_Loss of NN_1)
# @tf.function
# def f1(theta, phi):
#     # Write down the Jacobian matrix
#     t_u=(-(1-tf.cos(theta))*tf.cos(phi))
#     p_u=(-(1-tf.cos(theta))/tf.sin(theta)*tf.sin(phi))
#     t_v=(-(1-tf.cos(theta))*tf.sin(phi))
#     p_v=((1-tf.cos(theta))/tf.sin(theta)*tf.cos(phi))

#     h_0=u1(theta, phi)
#     h_t=tape.gradient(h_0, theta)[0]
#     h_p=tape.gradient(h_0, phi)[0]
#     # The Jacobian matrix multiplication
#     g_u=h_t*t_u + h_p*p_u
#     g_v=h_t*t_v + h_p*p_v

#     # The second partial derivatives
#     g_ut=tape.gradient(g_u, theta)[0]
#     g_up=tape.gradient(g_u, phi)[0]
#     g_vt=tape.gradient(g_v, theta)[0]
#     g_vp=tape.gradient(g_v, phi)[0]
#     # The Jacobian matrix multiplication
#     g_uu=g_ut*t_u + g_up*p_u
#     g_vv=g_vt*t_v + g_vp*p_v

#     # Calculate the residual equation
#     tru_ut=-trudtt(theta, phi)*(1-tf.cos(theta))*tf.cos(phi) - trudt(theta, phi)*tf.sin(theta)*tf.cos(phi) - trudpt(theta, phi)*((1-tf.cos(theta))/tf.sin(theta)*tf.sin(phi)) - trudp(theta, phi)*((tf.sin(theta)**2-tf.cos(theta)**2+tf.cos(theta))/(tf.sin(theta)**2)*tf.sin(phi))
#     tru_vt=-trudtt(theta, phi)*(1-tf.cos(theta))*tf.sin(phi) - trudt(theta, phi)*tf.sin(theta)*tf.sin(phi) - trudpt(theta, phi)*((1-tf.cos(theta))/tf.sin(theta)*tf.cos(phi)) - trudp(theta, phi)*((tf.sin(theta)**2-tf.cos(theta)**2+tf.cos(theta))/(tf.sin(theta)**2)*tf.cos(phi))
#     tru_up=-trudtp(theta, phi)*(1-tf.cos(theta))*tf.cos(phi) + trudt(theta, phi)*(1-tf.cos(theta))*tf.sin(phi) - trudpp(theta, phi)*(1-tf.cos(theta))/tf.sin(theta)*tf.sin(phi) - trudp(theta, phi)*((1-tf.cos(theta))/tf.sin(theta))*tf.cos(phi)
#     tru_vp=-trudtp(theta, phi)*(1-tf.cos(theta))*tf.sin(phi) - trudt(theta, phi)*(1-tf.cos(theta))*tf.cos(phi) - trudpp(theta, phi)*(1-tf.cos(theta))/tf.sin(theta)*tf.cos(phi) + trudp(theta, phi)*((1-tf.cos(theta))/tf.sin(theta))*tf.sin(phi)

#     F1=g_uu + g_vv - (tru_ut*t_u+tru_up*p_u + tru_vt*t_v + tru_vp*p_v)
#     retour = tf.reduce_mean(tf.square(F1))
#     return retour

# @tf.function
# def u2(theta, phi):
#     u2=model_2(tf.concat([theta, phi], axis=1))
#     return u2

# # Residual equation of NN_2(PDE_Loss of NN_2)
# @tf.function
# def f2(theta, phi):
#     # Write down the Jacobian matrix
#     t_u=(-(1-tf.cos(theta))*tf.cos(phi))
#     p_u=(-(1-tf.cos(theta))/tf.sin(theta)*tf.sin(phi))
#     t_v=(-(1-tf.cos(theta))*tf.sin(phi))
#     p_v=((1-tf.cos(theta))/tf.sin(theta)*tf.cos(phi))

#     h_0=u2(theta, phi)
#     h_t=tape.gradient(h_0, theta)[0]
#     h_p=tape.gradient(h_0, phi)[0]
#     # The Jacobian matrix multiplication
#     g_u=h_t*t_u + h_p*p_u
#     g_v=h_t*t_v + h_p*p_v

#     # The second partial derivatives
#     g_ut=tape.gradient(g_u, theta)[0]
#     g_up=tape.gradient(g_u, phi)[0]
#     g_vt=tape.gradient(g_v, theta)[0]
#     g_vp=tape.gradient(g_v, phi)[0]
#     # The Jacobian matrix multiplication
#     g_uu=g_ut*t_u + g_up*p_u
#     g_vv=g_vt*t_v + g_vp*p_v

#     # Calculate the residual equation
#     tru_ut=-trudtt(theta, phi)*(1-tf.cos(theta))*tf.cos(phi) - trudt(theta, phi)*tf.sin(theta)*tf.cos(phi) - trudpt(theta, phi)*((1-tf.cos(theta))/tf.sin(theta)*tf.sin(phi)) - trudp(theta, phi)*((tf.sin(theta)**2-tf.cos(theta)**2+tf.cos(theta))/(tf.sin(theta)**2)*tf.sin(phi))
#     tru_vt=-trudtt(theta, phi)*(1-tf.cos(theta))*tf.sin(phi) - trudt(theta, phi)*tf.sin(theta)*tf.sin(phi) - trudpt(theta, phi)*((1-tf.cos(theta))/tf.sin(theta)*tf.cos(phi)) - trudp(theta, phi)*((tf.sin(theta)**2-tf.cos(theta)**2+tf.cos(theta))/(tf.sin(theta)**2)*tf.cos(phi))
#     tru_up=-trudtp(theta, phi)*(1-tf.cos(theta))*tf.cos(phi) + trudt(theta, phi)*(1-tf.cos(theta))*tf.sin(phi) - trudpp(theta, phi)*(1-tf.cos(theta))/tf.sin(theta)*tf.sin(phi) - trudp(theta, phi)*((1-tf.cos(theta))/tf.sin(theta))*tf.cos(phi)
#     tru_vp=-trudtp(theta, phi)*(1-tf.cos(theta))*tf.sin(phi) - trudt(theta, phi)*(1-tf.cos(theta))*tf.cos(phi) - trudpp(theta, phi)*(1-tf.cos(theta))/tf.sin(theta)*tf.cos(phi) + trudp(theta, phi)*((1-tf.cos(theta))/tf.sin(theta))*tf.sin(phi)

#     F2=g_uu + g_vv - (tru_ut*t_u+tru_up*p_u + tru_vt*t_v + tru_vp*p_v)
#     retour = tf.reduce_mean(tf.square(F2))
#     return retour

# # Data_Loss of the neural networks
# @tf.function
# def mse(u1, u2):
#     return tf.reduce_mean(tf.square(u1 - u2))

# # ============================
# # Training begin
# # ============================
# loss = 0
# epochs = 10000
# opt1 = tf.keras.optimizers.Adam(learning_rate=1e-4)
# opt2 = tf.keras.optimizers.Adam(learning_rate=1e-4)
# epoch = 0
# loss_values = np.array([]) #total
# L1_values = np.array([]) #PDE loss of NN_1
# L2_values = np.array([]) #PDE loss of NN_2
# l_values = np.array([]) #Mse loss between NN_1 and NN_2 at boundary points

# # Record the start time of training NN_2
# start = time.time()
# # First, we need to train NN_2 first to get its approximation of true_u
# for epoch in range(epochs):
#     with tf.GradientTape(persistent=True) as tape:
#         T1_ = u1()
#         T2_ = u2()

#         # PDE loss of NN_1
#         L1 = 1 * f1()
#         # PDE loss of NN_2
#         L2 = 1 * f2( )
#         # Mse loss between NN_1 and NN_2 at boundary points
#         l = mse(T1_, T2_)
#         # Total Loss
#         loss = L1 + L2 + l

#     g1 = tape.gradient(loss, model_1.trainable_weights)
#     opt1.apply_gradients(zip(g1, model_1.trainable_weights))
#     g2 = tape.gradient(loss, model_2.trainable_weights)
#     opt2.apply_gradients(zip(g2, model_2.trainable_weights))

#     if epoch % 100 == 0 or epoch == epochs - 1:
#         print(f"{epoch:5}, {loss.numpy():.9f}")
#         loss_values = np.append(loss_values, loss)
#         L1_values = np.append(L1_values, L1)
#         L2_values = np.append(L2_values, L2)
#         l_values = np.append(l_values, l)

# # Record the end time of training NN_2
# end = time.time()
# computation_time_NN2 = {}
# computation_time_NN2["PINNs"] = end - start
# print(f"\ncomputation time of NNs: {end-start:.3f}\n")