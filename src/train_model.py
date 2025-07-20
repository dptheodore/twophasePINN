import os
import numpy as np
import tensorflow as tf
import pandas as pd
import math
from generate_points import get_training_data
import random
import logging
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)
import time

class TwoPhasePINNModel(tf.keras.Model):
    def __init__(self, base_model, rho1, rho2, mu1, mu2, sigma, g, U_ref, L_ref):
        super().__init__()
        self.base_model = base_model
        self.rho1, self.rho2 = rho1, rho2
        self.mu1, self.mu2 = mu1, mu2
        self.sigma, self.g = sigma, g
        self.U_ref, self.L_ref = U_ref, L_ref
        

    def call(self, inputs, training=False):
        x = tf.concat([inputs["x"], inputs["y"], inputs["t"]], axis=1)
        return self.base_model(x, training=training)

    def train_step(self, data):
        (a_batch, pde_batch, nsew_batch) = data
        (x_a, y_a) = a_batch
        (x_pde, y_pde) = pde_batch
        (x_nsew, y_nsew) = nsew_batch 

        with tf.GradientTape() as tape:
            output_tensors = self({"x": x_a[:,0:1], "y": x_a[:,1:2], "t": x_a[:,2:3]}, training=True)
            pred_nsew = self({"x": x_nsew[:,0:1], "y": x_nsew[:,1:2], "t": x_nsew[:,2:3]}, training=True)

            mse = tf.keras.losses.MeanSquaredError()
            #print(output_tensors[0][0])
            loss_a = tf.reduce_mean(tf.square(y_a - output_tensors[3]))
            loss_nsew = mse(y_nsew, pred_nsew[:,0:2])

            # PDE loss
            PDE_m, PDE_u, PDE_v, PDE_a = self.pde_caller(
                x_pde[:,0:1], x_pde[:,1:2], x_pde[:,2:3]
            )

            weights = tf.constant([1.0, 10.0, 10.0, 1.0], dtype=tf.float32)
            
            #compare PDE_m-a for this and master code
            #add in the f_PDE as the predicted

            y_pde = tf.transpose(y_pde)
            PDE_m = tf.transpose(PDE_m)
            PDE_u = tf.transpose(PDE_u)
            PDE_v = tf.transpose(PDE_v)
            PDE_a = tf.transpose(PDE_a)
            loss_PDE_m = mse(y_pde, PDE_m)
            loss_PDE_u = mse(y_pde, PDE_u)
            loss_PDE_v = mse(y_pde, PDE_v)
            loss_PDE_a = mse(y_pde, PDE_a)

            loss_PDE = tf.tensordot(tf.stack([loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_a]), np.array(weights), axes=1)

            total_loss = loss_a + loss_nsew + loss_PDE

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "loss": total_loss,
            "loss_a": loss_a,
            "loss_nsew": loss_nsew,
            "m": loss_PDE_m,
            "u": loss_PDE_u,
            "v": loss_PDE_v,
            "PDE_a": loss_PDE_a
        }

    def compute_gradients(self, x, y, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, t])
            inputs = {"x": x, "y": y, "t": t}
            out = self.call(inputs, training=True)
            u, v, p, a = out[:,0:1], out[:,1:2], out[:,2:3], out[:,3:4]

            # first derivatives
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            u_t = tape.gradient(u, t)

            v_x = tape.gradient(v, x)
            v_y = tape.gradient(v, y)
            v_t = tape.gradient(v, t)

            p_x = tape.gradient(p, x)
            p_y = tape.gradient(p, y)

            a_x = tape.gradient(a, x)
            a_y = tape.gradient(a, y)
            a_t = tape.gradient(a, t)

            # second derivatives
            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)

            v_xx = tape.gradient(v_x, x)
            v_yy = tape.gradient(v_y, y)

            a_xx = tape.gradient(a_x, x)
            a_yy = tape.gradient(a_y, y)
            a_xy = tape.gradient(a_x, y)

        del tape

        return (u, u_x, u_y, u_t, u_xx, u_yy,
                v, v_x, v_y, v_t, v_xx, v_yy,
                p, p_x, p_y,
                a, a_x, a_y, a_t, a_xx, a_yy, a_xy)

    def pde_caller(self, x, y, t):
        (u, u_x, u_y, u_t, u_xx, u_yy,
         v, v_x, v_y, v_t, v_xx, v_yy,
         p, p_x, p_y,
         a, a_x, a_y, a_t, a_xx, a_yy, a_xy) = self.compute_gradients(x, y, t)

        mu = self.mu2 + (self.mu1 - self.mu2) * a
        mu_x = (self.mu1 - self.mu2) * a_x
        mu_y = (self.mu1 - self.mu2) * a_y
        rho = self.rho2 + (self.rho1 - self.rho2) * a

        abs_grad_a = tf.sqrt(a_x**2 + a_y**2 + 1e-12)
        curvature = -((a_xx + a_yy)/abs_grad_a -
                     (a_x**2*a_xx + a_y**2*a_yy + 2*a_x*a_y*a_xy)/(abs_grad_a**3))

        rho_ref = self.rho2
        one_Re = mu/(rho_ref*self.U_ref*self.L_ref)
        one_Re_x = mu_x/(rho_ref*self.U_ref*self.L_ref)
        one_Re_y = mu_y/(rho_ref*self.U_ref*self.L_ref)
        one_We = self.sigma/(rho_ref*self.U_ref**2*self.L_ref)
        one_Fr = self.g*self.L_ref/self.U_ref**2

        PDE_m = u_x + v_y
        PDE_a = a_t + u*a_x + v*a_y
        PDE_u = (u_t + u*u_x + v*u_y)*rho/rho_ref + p_x \
                - one_We*curvature*a_x - one_Re*(u_xx + u_yy) \
                - 2.0*one_Re_x*u_x - one_Re_y*(u_y + v_x)
        PDE_v = (v_t + u*v_x + v*v_y)*rho/rho_ref + p_y \
                - one_We*curvature*a_y - rho/rho_ref*one_Fr \
                - one_Re*(v_xx + v_yy) - 2.0*one_Re_y*v_y - one_Re_x*(u_y + v_x)

        return PDE_m, PDE_u, PDE_v, PDE_a

def build_base_model(hidden_layers, output_dim):
    inputs = tf.keras.Input(shape=(3,))
    x = inputs
    for units in hidden_layers:
        x = tf.keras.layers.Dense(units, activation='tanh')(x)
    outputs = tf.keras.layers.Dense(output_dim)(x)
    return tf.keras.Model(inputs, outputs)

def df_to_dataset(df, feature_cols, target_cols, batch_size, shuffle=True):
    x = df[feature_cols].values.astype('float32')
    y = df[target_cols].values.astype('float32')
    if y.ndim == 1:
        y = y.reshape(-1,1)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def prepare_datasets(training_data, batch_size):
    ds_a = df_to_dataset(
        training_data['A'],
        feature_cols=['x_A', 'y_A', 't_A'],
        target_cols=['a_A'],
        batch_size=batch_size
    )
    ds_pde = df_to_dataset(
        training_data['PDE'],
        feature_cols=['x_PDE', 'y_PDE', 't_PDE'],
        target_cols=['f_PDE'],
        batch_size=batch_size
    )
    ds_nsew = df_to_dataset(
        training_data['NSEW'],
        feature_cols=['x_NSEW', 'y_NSEW', 't_NSEW'],
        target_cols=['u_NSEW', 'v_NSEW'],
        batch_size=batch_size
    )
    return tf.data.Dataset.zip((ds_a, ds_pde, ds_nsew))


if __name__ == "__main__":
    NOP_a = (500, 400)
    NOP_PDE = (400, 2000, 3000)
    NOP_north = (20, 20)
    NOP_south = (20, 20)
    NOP_east = (20, 20)
    NOP_west = (20, 20)

    tf.config.optimizer.set_jit(False)
    tf.config.run_functions_eagerly(True)

    training_data = get_training_data(NOP_a, NOP_PDE, NOP_north, NOP_south, NOP_east, NOP_west)

    #Starting physical values from paper
    mu = [1.0, 10.0]
    sigma = 24.5
    g = -0.98
    rho = [100, 1000]
    U_ref = 1.0
    L_ref = 0.25

    # Model
    hidden_layers = [350] * 8
    output_dim = 4  # u, v, p, a
    base_model = build_base_model(hidden_layers, output_dim)
    model = TwoPhasePINNModel(base_model, rho1=rho[0], rho2=rho[1], mu1=mu[0], mu2=mu[1], sigma=sigma, g=g, U_ref=U_ref, L_ref=L_ref)

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
    ]

    # Phased training
    learning_rates = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    batch_sizes = [9297, 9297, 9297, 9297, 9297] #i think this was the exact value the batch sizes were from the OG code
    epochs_per_phase = 5000

    for phase, (lr, bs) in enumerate(zip(learning_rates, batch_sizes), start=1):
        print(f"\n=== Phase {phase}/5 | LR: {lr:.1e} | BS: {bs} ===\n")

        # recompile with new optimizer to avoid .assign() issues
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer)

        dataset = prepare_datasets(training_data, batch_size=bs)

        epochs_this_phase = epochs_per_phase * phase
        start_epoch = epochs_per_phase * (phase - 1)

        for epoch in range(start_epoch + 1, epochs_this_phase + 1):
            start_time = time.time()
            dataset = prepare_datasets(training_data, batch_size=bs)
            print(dataset)
            epoch_losses = []
            for step, (a_batch, pde_batch, nsew_batch) in enumerate(dataset):
                batch_data = (a_batch, pde_batch, nsew_batch)
                print(batch_data)
                # call your `train_step()`, which already handles GradientTape
                losses = model.train_step(batch_data)

                epoch_losses.append(losses)

                # if you want to inspect output_tensors_A (or equivalent), you need to modify `train_step()` to return it
                # e.g. return losses, output_tensors_A

            # optionally log/aggregate losses here
            avg_loss = np.mean([l['loss'] for l in epoch_losses])

            # checkpoint if desired
            if epoch % 100 == 0:
                model.save_weights(f"checkpoint_phase{phase}_epoch{epoch}.h5")

            print(f"Epoch {epoch}/{epochs_this_phase} — Loss: {avg_loss:.6f} — Time: {time.time() - start_time:.2f}s")