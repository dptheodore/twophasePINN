import os
import numpy as np
import tensorflow as tf
import pandas as pd
import math
from generate_points import get_training_data

# Reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)

# === Model ===

class TwoPhasePINNModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def call(self, inputs, training=False):
        x = tf.concat([inputs["x"], inputs["y"], inputs["t"]], axis=1)
        return self.base_model(x, training=training)

    def train_step(self, data):
        (a_batch, pde_batch, nsew_batch) = data
        (x_a, y_a) = a_batch
        (x_pde, y_pde) = pde_batch
        (x_nsew, y_nsew) = nsew_batch

        with tf.GradientTape() as tape:
            pred_a = self({"x": x_a[:,0:1], "y": x_a[:,1:2], "t": x_a[:,2:3]}, training=True)
            pred_pde = self({"x": x_pde[:,0:1], "y": x_pde[:,1:2], "t": x_pde[:,2:3]}, training=True)
            pred_nsew = self({"x": x_nsew[:,0:1], "y": x_nsew[:,1:2], "t": x_nsew[:,2:3]}, training=True)

            # Dummy losses: adapt as needed
            loss_a = tf.reduce_mean(tf.square(pred_a[:, 3:4] - y_a))
            loss_pde = tf.reduce_mean(tf.square(pred_pde[:, 0:1] - y_pde))
            loss_bc = tf.reduce_mean(tf.square(pred_nsew[:, 0:2] - y_nsew))

            total_loss = loss_a + loss_pde + loss_bc

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "loss": total_loss,
            "loss_a": loss_a,
            "loss_pde": loss_pde,
            "loss_bc": loss_bc,
        }


# === Utilities ===

def build_base_model(hidden_layers, output_dim):
    inputs = tf.keras.Input(shape=(3,))
    x = inputs
    for units in hidden_layers:
        x = tf.keras.layers.Dense(units, activation='tanh')(x)
    outputs = tf.keras.layers.Dense(output_dim)(x)
    return tf.keras.Model(inputs, outputs)

def df_to_dataset(df, feature_cols, target_cols, batch_size=128, shuffle=True):
    x = df[feature_cols].values.astype('float32')
    y = df[target_cols].values.astype('float32')
    if y.ndim == 1:
        y = y.reshape(-1,1)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def prepare_datasets(training_data, batch_size=128):
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


# === Main ===

if __name__ == "__main__":
    # Data generation
    NOP_a = (500, 400)
    NOP_PDE = (400, 2000, 3000)
    NOP_north = (20, 20)
    NOP_south = (20, 20)
    NOP_east = (20, 20)
    NOP_west = (20, 20)

    training_data = get_training_data(NOP_a, NOP_PDE, NOP_north, NOP_south, NOP_east, NOP_west)

    # Model
    hidden_layers = [400] * 8
    output_dim = 4  # u, v, p, a
    base_model = build_base_model(hidden_layers, output_dim)
    model = TwoPhasePINNModel(base_model)

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
    batch_sizes = [128, 64, 64, 32, 32]
    epochs_per_phase = 5000

    for phase, (lr, bs) in enumerate(zip(learning_rates, batch_sizes), start=1):
        print(f"\n=== Phase {phase}/5 | LR: {lr:.1e} | BS: {bs} ===\n")
        model.optimizer.learning_rate.assign(lr)

        dataset = prepare_datasets(training_data, batch_size=bs)

        model.fit(
            dataset,
            epochs=epochs_per_phase * phase,
            initial_epoch=epochs_per_phase * (phase - 1),
            callbacks=callbacks
        )