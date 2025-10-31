import sys
sys.path.append("../utilities")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io
from generate_points_geom import get_training_data
from utilities import NNCreator, writeToJSONFile 
import time
import math
import glob
from datetime import datetime
import shutil
import logging
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects

TF_FUNCTION_SETTINGS = dict(reduce_retracing=True, experimental_relax_shapes=True)

# Define and register the custom sine activation function so Keras can find it by name
def sine_activation(x):
    return K.sin(x)
get_custom_objects().update({'sine': sine_activation})


# Set random seeds for reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)

class TwoPhasePinn(tf.keras.Model):
    """
    This class implements a physics-informed neural network in TensorFlow 2.
    It approximates the incompressible two-phase Navier-Stokes equations in 2D
    using a Volume-of-Fluid approach.
    """

    def __init__(self, hidden_layers, activation_functions, adaptive_activation_coeff,
                 adaptive_activation_n, adaptive_activation_init, use_ad_act,
                 loss_weights_PDE, mu, sigma, g, rho, u_ref, L_ref):
        super(TwoPhasePinn, self).__init__()

        # Physical Parameters
        self.mu1, self.mu2 = mu
        self.sigma = sigma
        self.g = g
        self.rho1, self.rho2 = rho
        self.U_ref = u_ref
        self.L_ref = L_ref
        self.rho_ref = self.rho2

        # Loss weights
        self.loss_weights_PDE = tf.constant(loss_weights_PDE, dtype=tf.float32)

        # Adaptive activation coefficients
        self.use_ad_act = use_ad_act
        self.ad_act_coeff = {}
        if self.use_ad_act:
            for key, initial_value in adaptive_activation_init.items():
                self.ad_act_coeff[key] = tf.Variable(initial_value, trainable=True, name=key, dtype=tf.float32)

        # --- Correctly build the model using NNCreator ---
        # 1. Create the activation function dictionary needed by NNCreator
        activation_functions_dict = self._get_activation_function_dict(
            hidden_layers, activation_functions, adaptive_activation_coeff, adaptive_activation_n
        )

        # 2. Define the output layer structure
        outputs = ["output_u", "output_v", "output_p", "output_a"]
        activations_output = [None, None, "exponential", "sigmoid"]
        output_layer = list(zip(outputs, activations_output))

        # 3. Instantiate NNCreator and build the model
        nn_creator = NNCreator(tf.float32)
        self.nn = nn_creator.get_model_dnn(3, hidden_layers, output_layer, activation_functions_dict, self.use_ad_act)


    def tf_median(self,x):
        x_sorted = tf.sort(tf.reshape(x, [-1]))
        n = tf.shape(x_sorted)[0]
        mid = n // 2
        return tf.cond(tf.equal(n % 2, 1),
                       lambda: tf.gather(x_sorted, mid),
                       lambda: 0.5 * (tf.gather(x_sorted, mid - 1) + tf.gather(x_sorted, mid)))

    # --- Diagnostics block ---
    def run_geom_diagnostics(self,
                             xg, yg, tg,        # inputs used for pred call
                             pred_a,            # predicted scalar field [N,1]
                             grad_x_pred, grad_y_pred,   # predicted grads [N,] or [N,1]
                             grad_x_gt=None, grad_y_gt=None,  # ground-truth grads (optional)
                             normal_x_gt=None, normal_y_gt=None,  # GT normals (optional)
                             loss_geom=None, loss_pde=None, loss_bc=None,
                             fd_eps=1e-4, fd_N=50):
        tf.print("=== GEOM DIAGNOSTICS ===")
    
        # squeeze shapes to be safe
        pred_a_s = tf.reshape(pred_a, [-1, 1])
        gx = tf.reshape(grad_x_pred, [-1])
        gy = tf.reshape(grad_y_pred, [-1])
    
        tf.print("shapes: pred_a", tf.shape(pred_a_s), "grad_x_pred", tf.shape(gx), "grad_y_pred", tf.shape(gy))
        tf.print("xg range:", tf.reduce_min(xg), tf.reduce_max(xg), "yg range:", tf.reduce_min(yg), tf.reduce_max(yg))
    
        # pred_a stats
        tf.print("pred_a mean/std:", tf.reduce_mean(pred_a_s), tf.math.reduce_std(pred_a_s))
    
        # --- 1) Finite difference vs autodiff (sample subset) ---
        N = tf.shape(pred_a_s)[0]
        fd_N = tf.minimum(fd_N, N)
        idx = tf.random.shuffle(tf.range(N))[:fd_N]
    
        x_sel = tf.gather(xg, idx)
        y_sel = tf.gather(yg, idx)
        t_sel = tf.gather(tg, idx) if tg is not None else tf.zeros_like(x_sel)
    
        eps = tf.constant(fd_eps, dtype=x_sel.dtype)
    
        inp_center = tf.concat([x_sel, y_sel, t_sel], axis=1)
        inp_xp = tf.concat([x_sel + eps, y_sel, t_sel], axis=1)
        inp_xn = tf.concat([x_sel - eps, y_sel, t_sel], axis=1)
        inp_yp = tf.concat([x_sel, y_sel + eps, t_sel], axis=1)
        inp_yn = tf.concat([x_sel, y_sel - eps, t_sel], axis=1)
    
        # evaluate model (assumes self.call returns outputs and pred_a is last)
        pa_center = self.call(inp_center)[-1]
        pa_xp = self.call(inp_xp)[-1]
        pa_xn = self.call(inp_xn)[-1]
        pa_yp = self.call(inp_yp)[-1]
        pa_yn = self.call(inp_yn)[-1]
    
        fd_gx = tf.reshape((pa_xp - pa_xn) / (2.0 * eps), [-1])
        fd_gy = tf.reshape((pa_yp - pa_yn) / (2.0 * eps), [-1])
    
        # recompute AD grads at sample points (batch_jacobian)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x_sel, y_sel])
            pa = self.call(tf.concat([x_sel, y_sel, t_sel], axis=1))[-1]  # pred_a at selected points

        ad_gx = tf.squeeze(tape2.batch_jacobian(pa, x_sel), -1)
        ad_gy = tf.squeeze(tape2.batch_jacobian(pa, y_sel), -1)
        
        del tape2
    
        df_gx = fd_gx - ad_gx
        df_gy = fd_gy - ad_gy
    
        tf.print("FD vs AD grad_x mean/absmax:", tf.reduce_mean(df_gx), tf.reduce_max(tf.abs(df_gx)))
        tf.print("FD vs AD grad_y mean/absmax:", tf.reduce_mean(df_gy), tf.reduce_max(tf.abs(df_gy)))
    
        # --- 2) Normal magnitudes and orientation (if GT provided) ---
        norms = tf.norm(tf.stack([gx, gy], axis=1), axis=1)  # per-point norm
        tf.print("pred normal norm mean/std/min/max:", tf.reduce_mean(norms), tf.math.reduce_std(norms),
                 tf.reduce_min(norms), tf.reduce_max(norms))
        # unit normals (guard small norms)
        unit_pred = tf.stack([gx, gy], axis=1) / (tf.expand_dims(norms, -1) + 1e-12)
    
        if normal_x_gt is not None and normal_y_gt is not None:
            normal_gt = tf.stack([tf.reshape(normal_x_gt, [-1]), tf.reshape(normal_y_gt, [-1])], axis=1)
            gt_norms = tf.norm(normal_gt, axis=1)
            unit_gt = normal_gt / (tf.expand_dims(gt_norms, -1) + 1e-12)
    
            # cosine and angles
            cosang = tf.reduce_sum(unit_pred * unit_gt, axis=1)
            cosang = tf.clip_by_value(cosang, -1.0, 1.0)
            ang = tf.acos(cosang) * 180.0 / 3.141592653589793
            tf.print("normal angle deg mean/median/max:", tf.reduce_mean(ang), self.tf_median(ang), tf.reduce_max(ang))
    
            # also print fraction with angle>90 (sign flip)
            n_flip = tf.reduce_sum(tf.cast(ang > 90.0, tf.float32))
            tf.print("fraction of points with angle>90deg (possible flip):", n_flip / tf.cast(tf.shape(ang)[0], tf.float32))
    
        # --- 3) Aggregated grad_vec compare (raw sum) ---
        gv_x_pred = tf.reduce_sum(gx)
        gv_y_pred = tf.reduce_sum(gy)
        tf.print("aggregated grad_vec_pred (sum):", gv_x_pred, gv_y_pred)
    
        if (grad_x_gt is not None) and (grad_y_gt is not None):
            gv_x_gt = tf.reduce_sum(tf.reshape(grad_x_gt, [-1]))
            gv_y_gt = tf.reduce_sum(tf.reshape(grad_y_gt, [-1]))
            tf.print("aggregated grad_vec_gt  (sum):", gv_x_gt, gv_y_gt)
            rel_err = tf.norm(tf.stack([gv_x_pred - gv_x_gt, gv_y_pred - gv_y_gt])) / (tf.norm(tf.stack([gv_x_gt, gv_y_gt])) + 1e-12)
            tf.print("relative error (sum):", rel_err)
    
        # --- 4) Soft-contour weighted aggregated grad_vec ---
        eps_soft = tf.constant(0.05, dtype=pred_a_s.dtype)  # tune this
        weight = tf.exp(-tf.square(pred_a_s) / (2.0 * eps_soft * eps_soft))
        gv_x_pred_w = tf.reduce_sum(tf.reshape(weight, [-1]) * gx)
        gv_y_pred_w = tf.reduce_sum(tf.reshape(weight, [-1]) * gy)
        tf.print("weighted aggregated grad_vec_pred:", gv_x_pred_w, gv_y_pred_w)
        if (grad_x_gt is not None) and (grad_y_gt is not None):
            # optional: weight GT in same way if GT is per-point
            gv_x_gt_w = tf.reduce_sum(tf.reshape(weight, [-1]) * tf.reshape(grad_x_gt, [-1]))
            gv_y_gt_w = tf.reduce_sum(tf.reshape(weight, [-1]) * tf.reshape(grad_y_gt, [-1]))
            tf.print("weighted aggregated grad_vec_gt :", gv_x_gt_w, gv_y_gt_w)
            rel_err_w = tf.norm(tf.stack([gv_x_pred_w - gv_x_gt_w, gv_y_pred_w - gv_y_gt_w])) / (tf.norm(tf.stack([gv_x_gt_w, gv_y_gt_w])) + 1e-12)
            tf.print("relative error (weighted):", rel_err_w)
    
        # --- 5) Per-point errors and top offenders (if GT grads available) ---
        if (grad_x_gt is not None) and (grad_y_gt is not None):
            per_err = tf.sqrt((gx - tf.reshape(grad_x_gt, [-1]))**2 + (gy - tf.reshape(grad_y_gt, [-1]))**2)
            tf.print("per_err mean/median/max:", tf.reduce_mean(per_err), self.tf_median(per_err), tf.reduce_max(per_err))
            k = tf.minimum(10, tf.shape(per_err)[0])
            topk = tf.math.top_k(per_err, k=k)
            tf.print("top per_err values:", topk.values)
            tf.print("top per_err idx:", topk.indices)
            coords_top = tf.gather(tf.concat([tf.reshape(xg, [-1,1]), tf.reshape(yg, [-1,1])], axis=1), topk.indices)
            tf.print("coords of top offenders (x,y):", coords_top)
    
        # --- 6) Loss magnitudes if provided ---
        if loss_geom is not None:
            tf.print("loss_geom:", loss_geom)
        if loss_pde is not None:
            tf.print("loss_pde:", loss_pde)
        if loss_bc is not None:
            tf.print("loss_bc:", loss_bc)
    
        tf.print("=== END DIAGNOSTICS ===")

    def debug_geom_visual(self, data_GEOM):
        import matplotlib.pyplot as plt
        import numpy as np

        xg, yg, tg, grad_x_gt, grad_y_gt, normal_x_gt, normal_y_gt = data_GEOM

        # Run model to get predicted gradients
        delta = 0.5 * 0.00390625  # or whatever your grid spacing is
        _, _, _, a_left   = self.call(tf.concat([xg - delta, yg, tg], axis=1))
        _, _, _, a_right  = self.call(tf.concat([xg + delta, yg, tg], axis=1))
        _, _, _, a_bottom = self.call(tf.concat([xg, yg - delta, tg], axis=1))
        _, _, _, a_top    = self.call(tf.concat([xg, yg + delta, tg], axis=1))

        grad_x_pred = (a_right - a_left).numpy().squeeze()
        grad_y_pred = (a_top - a_bottom).numpy().squeeze()

        grad_x_gt = grad_x_gt.numpy().squeeze()
        grad_y_gt = grad_y_gt.numpy().squeeze()
        xg = xg.numpy().squeeze()
        yg = yg.numpy().squeeze()

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.quiver(xg, yg, grad_x_gt, grad_y_gt, angles="xy", scale_units="xy", scale=1)
        plt.title("Ground Truth Grad Vecs")

        plt.subplot(1, 2, 2)
        plt.quiver(xg, yg, grad_x_pred, grad_y_pred, angles="xy", scale_units="xy", scale=1)
        plt.title("Predicted Grad Vecs")

        plt.gca().invert_yaxis()  # optional if your Y goes downward
        plt.show()



    # def debug_geom_visual(xg, yg, grad_vec_pred, grad_vec_gt, stride=4, title="Geom Debug"):
    #     """
    #     Visualize predicted vs ground-truth gradient fields.
    #     Can be called from within your geom loss computation for debugging.

    #     Args:
    #         xg, yg: 1D or 2D arrays of coordinates (used for plotting).
    #         grad_vec_pred: [H, W, 2] predicted gradients (Tensor or np array).
    #         grad_vec_gt:   [H, W, 2] ground-truth gradients (Tensor or np array).
    #         stride: int, subsampling step for quiver plot.
    #         title: figure title.
    #     """
    #     # --- Convert tensors to numpy ---
    #     if hasattr(grad_vec_pred, "numpy"):
    #         grad_vec_pred = grad_vec_pred.numpy()
    #     if hasattr(grad_vec_gt, "numpy"):
    #         grad_vec_gt = grad_vec_gt.numpy()

    #     # --- Compute magnitudes and normalized directions ---
    #     grad_mag_pred = np.linalg.norm(grad_vec_pred, axis=-1)
    #     grad_mag_gt = np.linalg.norm(grad_vec_gt, axis=-1)
    #     eps = 1e-8
    #     grad_dir_pred = grad_vec_pred / (grad_mag_pred[..., None] + eps)
    #     grad_dir_gt = grad_vec_gt / (grad_mag_gt[..., None] + eps)

    #     # --- Build coordinate grid for quiver ---
    #     if xg.ndim == 1 and yg.ndim == 1:
    #         X, Y = np.meshgrid(xg, yg)
    #     else:
    #         X, Y = xg, yg

    #     # --- Make figure ---
    #     fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    #     fig.suptitle(title, fontsize=14)

    #     # Magnitude maps
    #     axs[0,0].imshow(grad_mag_gt, cmap='viridis')
    #     axs[0,0].set_title("GT |∇φ|")

    #     axs[0,1].imshow(grad_mag_pred, cmap='viridis')
    #     axs[0,1].set_title("Pred |∇φ|")

    #     axs[0,2].imshow(np.abs(grad_mag_pred - grad_mag_gt), cmap='inferno')
    #     axs[0,2].set_title("Abs diff |∇φ|")

    #     # Direction fields
    #     axs[1,0].quiver(X[::stride,::stride], Y[::stride,::stride],
    #                     grad_dir_gt[::stride,::stride,0],
    #                     -grad_dir_gt[::stride,::stride,1], color='cyan', scale=30)
    #     axs[1,0].set_title("GT ∇φ direction")

    #     axs[1,1].quiver(X[::stride,::stride], Y[::stride,::stride],
    #                     grad_dir_pred[::stride,::stride,0],
    #                     -grad_dir_pred[::stride,::stride,1], color='orange', scale=30)
    #     axs[1,1].set_title("Pred ∇φ direction")

    #     dir_diff = np.sum((grad_dir_pred - grad_dir_gt)**2, axis=-1)
    #     axs[1,2].imshow(dir_diff, cmap='magma')
    #     axs[1,2].set_title("Direction mismatch")

    #     for ax in axs.flat:
    #         ax.axis("off")

    #     plt.tight_layout()
    #     plt.show()


    def compute_geom_loss(self, data_GEOM, geom_weight=1e1, delta=0.5 * 0.00390625,
                          eps_soft=0.1, min_weight=0.05, huber_delta=1e-3, debug=False):
        """
        Stabilized geometric loss:
        - uses central-difference divided by 2*delta (correct derivative scale)
        - compares magnitudes for grad_vec (use abs if grad_vec is a length)
        - normalizes pred and GT normals before angular comparison
        - uses Huber loss (robust) instead of raw MSE
        - geom_weight should start small (1e-3..1e-2) and be ramped up in training
        """
        xg, yg, tg, grad_x_gt, grad_y_gt, normal_x_gt, normal_y_gt = data_GEOM

        # ---- 1) Evaluate predicted a at offset positions ----
        pts_left   = tf.concat([xg - delta, yg, tg], axis=1)
        pts_right  = tf.concat([xg + delta, yg, tg], axis=1)
        pts_bottom = tf.concat([xg, yg - delta, tg], axis=1)
        pts_top    = tf.concat([xg, yg + delta, tg], axis=1)

        # inference (training mode default keeps BN if any; you can force training=False if desired)
        _, _, _, a_left   = self(pts_left, training=False)
        _, _, _, a_right  = self(pts_right, training=False)
        _, _, _, a_bottom = self(pts_bottom, training=False)
        _, _, _, a_top    = self(pts_top, training=False)

        # ---- 2) Central-difference derivative estimate (correct scale) ----
        # d a / dx ≈ (a(x+delta) - a(x-delta)) / (2*delta)
        denom = tf.cast(2.0 * delta, tf.float32)
        grad_x_pred = (a_right - a_left) / denom
        grad_y_pred = (a_top - a_bottom) / denom

        # Produce grad_vec_pred as magnitudes if GT is length-like; use abs if appropriate
        # If your grad_vec_gt are positive lengths, compare magnitudes:
        grad_vec_pred_mag = tf.concat([tf.abs(grad_x_pred), tf.abs(grad_y_pred)], axis=1)

        # ---- 3) Predicted normal ----
        # Use derivative field for normal. Guard tiny norms with epsilon.
        norm_mag = tf.sqrt(tf.square(grad_x_pred) + tf.square(grad_y_pred) + 1e-8)
        normal_pred = tf.concat([-grad_x_pred / norm_mag, -grad_y_pred / norm_mag], axis=1)

        # ---- 4) Soft-contour weighting (use clipped a_center for stability) ----
        _, _, _, a_center = self(tf.concat([xg, yg, tg], axis=1), training=False)
        a_center = tf.clip_by_value(a_center, 0.0, 1.0)  # ensure 0..1
        weight = tf.exp(-tf.square(a_center) / (2.0 * eps_soft**2))
        weight = tf.reshape(weight, [-1])
        weight = tf.maximum(weight, min_weight)
        w2 = tf.expand_dims(weight, axis=1)

        # ---- 5) Prepare GT and normalization ----
        grad_x_gt_s = tf.squeeze(grad_x_gt, -1)
        grad_y_gt_s = tf.squeeze(grad_y_gt, -1)
        grad_vec_gt_mag = tf.concat([grad_x_gt_s[:, None], grad_y_gt_s[:, None]], axis=1)

        # Normalize grad vectors to comparable scale if GTs are much larger/smaller:
        # compute median magnitude and use it to normalize (robust)
        pred_mag = tf.sqrt(tf.reduce_sum(tf.square(grad_vec_pred_mag), axis=1, keepdims=True)) + 1e-12
        gt_mag = tf.sqrt(tf.reduce_sum(tf.square(grad_vec_gt_mag), axis=1, keepdims=True)) + 1e-12

        # optionally normalize per-point to unit magnitude before direction comparison,
        # but for lengths we want to compare magnitudes so we skip direction normalization here.

        # ---- 6) Huber loss on grad magnitudes (robust to outliers) ----
        # Huber: 1/2*x^2 if |x|<=d else d*(|x|-0.5*d)
        diff_grad = grad_vec_pred_mag - grad_vec_gt_mag
        abs_diff = tf.abs(diff_grad)
        huber_mask = tf.cast(abs_diff <= huber_delta, tf.float32)
        loss_grad_vec_point = huber_mask * 0.5 * tf.square(diff_grad) + (1.0 - huber_mask) * (huber_delta * (abs_diff - 0.5 * huber_delta))
        loss_grad_vec = tf.reduce_mean(w2 * tf.reduce_sum(loss_grad_vec_point, axis=1, keepdims=True))

        # ---- 7) Normal (direction) loss: use cosine distance (robust) ----
        normal_x_gt_s = tf.squeeze(normal_x_gt, -1)
        normal_y_gt_s = tf.squeeze(normal_y_gt, -1)
        normal_gt = tf.stack([normal_x_gt_s, normal_y_gt_s], axis=1)
        # normalize GT normals
        gt_norms = tf.sqrt(tf.reduce_sum(tf.square(normal_gt), axis=1, keepdims=True)) + 1e-12
        normal_gt_unit = normal_gt / gt_norms
        # clip pred to avoid NaNs
        normal_pred_unit = normal_pred / (tf.sqrt(tf.reduce_sum(tf.square(normal_pred), axis=1, keepdims=True)) + 1e-12)

        # cosine similarity -> turn into a distance in [0,2]
        cos_sim = tf.reduce_sum(normal_pred_unit * normal_gt_unit, axis=1, keepdims=True)
        cos_sim = tf.clip_by_value(cos_sim, -1.0, 1.0)
        # angle distance (robust): 1 - cos_sim is fine and differentiable
        loss_normal_point = 1.0 - cos_sim
        loss_normal = tf.reduce_mean(w2 * loss_normal_point)

        # ---- 8) Combine with geom_weight (keep geom_weight small initially) ----
        loss_geom = geom_weight * (loss_grad_vec + loss_normal)


        return loss_geom

    def _get_activation_function_dict(self, hidden_layers, activation_functions, adaptive_activation_coeff, adaptive_activation_n):
        """Helper to create the activation dictionary for NNCreator."""
        activation_dict = {i: [None, None, 0] for i in range(1, len(hidden_layers) + 1)}
        for layer_no in activation_dict:
            activation_dict[layer_no][2] = adaptive_activation_n[layer_no - 1]
            for func_name, layers in activation_functions.items():
                if layer_no in layers:
                    activation_dict[layer_no][0] = func_name
            if self.use_ad_act:
                for coeff_name, layers in adaptive_activation_coeff.items():
                    if layer_no in layers:
                        activation_dict[layer_no][1] = self.ad_act_coeff[coeff_name]
        return activation_dict

    def call(self, inputs, training=True):
        # The model built by NNCreator is now stored in self.nn
        return self.nn(inputs)

    @tf.function(**TF_FUNCTION_SETTINGS)
    def compute_gradients(self, x, y, t):
        # Use a nested tape to compute second-order derivatives
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y, t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y, t])
                u, v, p, a = self.call(tf.concat([x, y, t], axis=1))

            # First-order gradients computed with the inner tape
            u_x = tape1.gradient(u, x)
            u_y = tape1.gradient(u, y)
            u_t = tape1.gradient(u, t)
            v_x = tape1.gradient(v, x)
            v_y = tape1.gradient(v, y)
            v_t = tape1.gradient(v, t)
            p_x = tape1.gradient(p, x)
            p_y = tape1.gradient(p, y)
            a_x = tape1.gradient(a, x)
            a_y = tape1.gradient(a, y)
            a_t = tape1.gradient(a, t)

        # Second-order gradients computed with the outer tape
        u_xx = tape2.gradient(u_x, x)
        u_yy = tape2.gradient(u_y, y)
        v_xx = tape2.gradient(v_x, x)
        v_yy = tape2.gradient(v_y, y)
        a_xx = tape2.gradient(a_x, x)
        a_yy = tape2.gradient(a_y, y)
        a_xy = tape2.gradient(a_x, y)

        # Clean up tapes
        del tape1, tape2

        return (u, u_x, u_y, u_t, u_xx, u_yy), \
               (v, v_x, v_y, v_t, v_xx, v_yy), \
               (p, p_x, p_y), \
               (a, a_x, a_y, a_t, a_xx, a_yy, a_xy)

    @tf.function(**TF_FUNCTION_SETTINGS)
    def PDE_caller(self, x, y, t):
        u_grads, v_grads, p_grads, a_grads = self.compute_gradients(x, y, t)
        u, u_x, u_y, u_t, u_xx, u_yy = u_grads
        v, v_x, v_y, v_t, v_xx, v_yy = v_grads
        p, p_x, p_y = p_grads
        a, a_x, a_y, a_t, a_xx, a_yy, a_xy = a_grads

        mu = self.mu2 + (self.mu1 - self.mu2) * a
        mu_x = (self.mu1 - self.mu2) * a_x
        mu_y = (self.mu1 - self.mu2) * a_y
        rho = self.rho2 + (self.rho1 - self.rho2) * a

        abs_interface_grad = tf.sqrt(a_x**2 + a_y**2 + np.finfo(float).eps)
        curvature = -((a_xx + a_yy) / abs_interface_grad -
                      (a_x**2 * a_xx + a_y**2 * a_yy + 2 * a_x * a_y * a_xy) / tf.pow(abs_interface_grad, 3))

        one_Re = mu / (self.rho_ref * self.U_ref * self.L_ref)
        one_Re_x = mu_x / (self.rho_ref * self.U_ref * self.L_ref)
        one_Re_y = mu_y / (self.rho_ref * self.U_ref * self.L_ref)
        one_We = self.sigma / (self.rho_ref * self.U_ref**2 * self.L_ref)
        one_Fr = self.g * self.L_ref / self.U_ref**2

        PDE_m = u_x + v_y
        PDE_a = a_t + u * a_x + v * a_y
        PDE_u = (u_t + u * u_x + v * u_y) * rho / self.rho_ref + p_x - \
                one_We * curvature * a_x - one_Re * (u_xx + u_yy) - \
                2.0 * one_Re_x * u_x - one_Re_y * (u_y + v_x)
        PDE_v = (v_t + u * v_x + v * v_y) * rho / self.rho_ref + p_y - \
                one_We * curvature * a_y - one_Re * (v_xx + v_yy) - \
                rho / self.rho_ref * one_Fr - 2.0 * one_Re_y * v_y - one_Re_x * (u_y + v_x)

        return PDE_m, PDE_u, PDE_v, PDE_a

    @tf.function(**TF_FUNCTION_SETTINGS)
    def compute_loss(self, data_A, data_PDE, data_N, data_EW, data_NSEW, data_GEOM):
        # Unpack tensor tuples
        x_A, y_A, t_A, a_A = data_A
        x_PDE, y_PDE, t_PDE = data_PDE
        x_N, y_N, t_N, p_N = data_N
        x_E, y_E, t_EW, x_W, y_W = data_EW
        x_NSEW, y_NSEW, t_NSEW, u_NSEW, v_NSEW = data_NSEW

        f_PDE = tf.zeros_like(x_PDE)

        # Loss A (Volume Fraction)
        output_tensors = self.call(tf.concat([x_A, y_A, t_A], axis=1))
        loss_a_A = tf.reduce_mean(tf.square(a_A - output_tensors[3]))

        # Loss NSEW (Boundary Conditions)
        pred_u_NSEW, pred_v_NSEW, _, _ = self.call(tf.concat([x_NSEW, y_NSEW, t_NSEW], axis=1))
        loss_u_NSEW = tf.reduce_mean(tf.square(u_NSEW - pred_u_NSEW))
        loss_v_NSEW = tf.reduce_mean(tf.square(v_NSEW - pred_v_NSEW))

        # Loss N (Pressure at North boundary)
        _, _, pred_p_N, _ = self.call(tf.concat([x_N, y_N, t_N], axis=1))
        loss_p_N = tf.reduce_mean(tf.square(p_N - pred_p_N))

        # Loss EW (Periodic Boundary)
        pred_east = self.call(tf.concat([x_E, y_E, t_EW], axis=1))
        pred_west = self.call(tf.concat([x_W, y_W, t_EW], axis=1))
        loss_u_EW = tf.reduce_mean(tf.square(pred_east[0] - pred_west[0]))
        loss_v_EW = tf.reduce_mean(tf.square(pred_east[1] - pred_west[1]))
        loss_p_EW = tf.reduce_mean(tf.square(pred_east[2] - pred_west[2]))

        loss_BC = loss_u_NSEW + loss_v_NSEW + loss_p_N + loss_u_EW + loss_v_EW + loss_p_EW

        # Loss PDE (Physics-Informed)
        PDE_m, PDE_u, PDE_v, PDE_a = self.PDE_caller(x_PDE, y_PDE, t_PDE)
        loss_PDE_m = tf.reduce_mean(tf.square(f_PDE - PDE_m))
        loss_PDE_u = tf.reduce_mean(tf.square(f_PDE - PDE_u))
        loss_PDE_v = tf.reduce_mean(tf.square(f_PDE - PDE_v))
        loss_PDE_a = tf.reduce_mean(tf.square(f_PDE - PDE_a))

        loss_PDE = tf.tensordot(tf.stack([loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_a]), self.loss_weights_PDE, 1)

        loss_geom = self.compute_geom_loss(data_GEOM, loss_PDE, loss_BC, debug=False)

        # === Combine all losses ===
        total_loss = loss_a_A + loss_BC + loss_PDE + loss_geom

        return total_loss, loss_a_A, loss_BC, loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_a, loss_geom


    @tf.function(**TF_FUNCTION_SETTINGS)
    def train_step(self, optimizer, data_A, data_PDE, data_N, data_EW, data_NSEW, data_GEOM):
        with tf.GradientTape() as tape:
            # Pass the tensor tuples directly to compute_loss
            losses = self.compute_loss(data_A, data_PDE, data_N, data_EW, data_NSEW, data_GEOM)
            total_loss = losses[0]

        gradients = tape.gradient(total_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return losses


def setup_output_directory():
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    dirname = os.path.abspath(os.path.join("checkpoints", datetime.now().strftime("%b-%d-%Y_%H-%M-%S")))
    os.mkdir(dirname)
    
    shutil.copyfile(__file__, os.path.join(dirname, os.path.basename(__file__)))
    if os.path.exists("generate_points.py"):
        shutil.copyfile("generate_points.py", os.path.join(dirname, "generate_points.py"))
    
    logpath = os.path.join(dirname, "output.log")
    return dirname, logpath

def get_logger(logpath):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)
    
    fh = logging.FileHandler(logpath)
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    
    return logger

def get_proportional_batch_sizes(total_batch_size, training_data, logger):
    """Calculates proportional batch sizes for each dataset."""
    num_samples_total = sum(len(df) for df in training_data.values())
    num_batches = math.ceil(num_samples_total / total_batch_size)
    
    batch_sizes = {}
    for key, df in training_data.items():
        if len(df) > 0:
            proportion = len(df) / num_samples_total
            batch_sizes[key] = math.ceil(proportion * total_batch_size)
        else:
            batch_sizes[key] = 0

    batch_sizes = {
        'A': max(16, batch_sizes['A']),
        'PDE': max(32, batch_sizes['PDE']),
        'N': max(16, batch_sizes['N']),
        'EW': max(16, batch_sizes['EW']),
        'NSEW': max(16, batch_sizes['NSEW']),
        'GEOM': max(16, batch_sizes['GEOM'])
    }
            
    logger.info(f"Total samples: {num_samples_total}, Desired batch size: {total_batch_size}")
    logger.info(f"Calculated num_batches: {num_batches}, Proportional batch sizes: {batch_sizes}")
    return batch_sizes, num_batches

def main():
    """
    This script trains a PINN for the rising bubble case.
    """
    dirname, logpath = setup_output_directory()
    logger = get_logger(logpath)

    NOP_a = (500, 400)
    NOP_PDE = (400, 2000, 3000)
    NOP_north = (20, 20)
    NOP_south = (20, 20)
    NOP_east = (20, 20)
    NOP_west = (20, 20)

    #every point not in boundary, compute small marching squares window around it (i.e. 1 voxel)
    #plot is bunch of little squares around each little point in domain, overlapping squares everywhere, espec. near interface
    #once have meshes can compute theorems
    #theorem is compute fraction of top and bottom and fraction of left and right that are immersed inside the red or blue fields
    #gradient, deriv of volume fraciton field, interp on mesh, --> gets normal vec
    #4 points vert 5 points horiz entire domain, compute theorems in each region
    #eventually nice: for any arb point in domain compute region around it
    training_data = get_training_data(NOP_a, NOP_PDE, NOP_north, NOP_south, NOP_east, NOP_west)
    #training_data['A']['a_A'] = (training_data['A']['a_A'] * 2.0) - 1.0

    # --- NN Architecture and Hyperparameters --- #
    no_layers = 8
    hidden_layers = [400] * no_layers

    # --- CHOOSE YOUR CONFIGURATION ---
    activation_choice = 'tanh'  # Options: 'tanh' or 'sine'
    use_aac_1 = False
    use_aac_2 = False #If you declare both false, fixed/no activation is used

    # Build activation function dictionary based on choice
    activation_functions = {activation_choice: range(1, no_layers + 1)}

    # Determine adaptive activation mode for file naming and logic
    adaptive_mode = 'fixed'
    if use_aac_1:
        adaptive_mode = 'activation1'
    elif use_aac_2:
        adaptive_mode = 'activation2'
    
    logger.info(f"Configuration: Activation='{activation_choice}', Mode='{adaptive_mode}'")
    
    # Conditionally define the adaptive activation parameters
    if use_aac_1:
        adaptive_activation_coeff = {"aac_1": range(1, no_layers + 1)}
        adaptive_activation_init = {"aac_1": 0.1}
        adaptive_activation_n = [10] * no_layers
    elif use_aac_2:
        adaptive_activation_coeff = {
            "aac_2_a": range(1, 5),
            "aac_2_b": range(5, 9)
        }
        adaptive_activation_init = {
            "aac_2_a": 0.05,
            "aac_2_b": 0.1
        }
        adaptive_activation_n = [20] * 4 + [10] * 4
    else: # 'fixed' mode
        adaptive_activation_coeff = {}
        adaptive_activation_init = {}
        adaptive_activation_n = [10] * no_layers

    use_adaptive_activation = use_aac_1 or use_aac_2
    
    mu = [1.0, 10.0]
    sigma = 24.5
    g = -0.98
    rho = [100, 1000]
    u_ref = 1.0
    L_ref = 0.25
    loss_weights_PDE = [1.0, 10.0, 10.0, 1.0]
    epochs_list = [5000] * 5
    learning_rates = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    checkpoint_interval = 1
    num_of_batches = 20
    
    num_samples_total = sum(len(df) for df in training_data.values())
    total_batch_size = math.ceil(num_samples_total / num_of_batches) 

    pinn = TwoPhasePinn(hidden_layers, activation_functions, adaptive_activation_coeff,
                      adaptive_activation_n, adaptive_activation_init, use_adaptive_activation,
                      loss_weights_PDE, mu, sigma, g, rho, u_ref, L_ref)

    pinn.nn.load_weights('initial_weights.h5')

    start_total = time.time()
    
    history_loss_a = []
    history_loss_f_uv = []
    history_loss_f_ma = []
    history_loss_geom = []
    
    def to_tensor_tuple(df, columns):
        return tuple(tf.constant(df[c].to_numpy().reshape(-1, 1), dtype=tf.float32) for c in columns)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates[0])
    current_best_total_loss = float('inf')
    for i, (epochs, lr) in enumerate(zip(epochs_list, learning_rates)):
        logger.info(f"\n--- Starting Training Phase {i+1}/{len(epochs_list)} ---")
        logger.info(f"Epochs: {epochs}, Learning Rate: {lr}")
        optimizer.learning_rate.assign(lr)
        prop_batch_sizes, num_batches = get_proportional_batch_sizes(total_batch_size, training_data, logger)
        start_checkpoint_time = time.time()
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            shuffled_data = {key: df.sample(frac=1) for key, df in training_data.items()}

            for b in range(num_batches):
                batch_dict = {}
                for key, df in shuffled_data.items():
                    start_idx = b * prop_batch_sizes[key]
                    end_idx = (b + 1) * prop_batch_sizes[key]
                    batch_dict[key] = df.iloc[start_idx:end_idx]

                if all(batch.empty for batch in batch_dict.values()):
                    continue

                data_A = to_tensor_tuple(batch_dict['A'], batch_dict['A'].columns)
                data_PDE = to_tensor_tuple(batch_dict['PDE'], ['x_PDE', 'y_PDE', 't_PDE'])
                data_N = to_tensor_tuple(batch_dict['N'], batch_dict['N'].columns)
                data_EW = to_tensor_tuple(batch_dict['EW'], ['x_E', 'y_E', 't_EW', 'x_W', 'y_W'])
                data_NSEW = to_tensor_tuple(batch_dict['NSEW'], batch_dict['NSEW'].columns)
                data_GEOM = to_tensor_tuple(batch_dict['GEOM'], batch_dict['GEOM'].columns)
                pinn.debug_geom_visual(data_GEOM)
                batch_loss_values = pinn.train_step(optimizer, data_A, data_PDE, data_N, data_EW, data_NSEW, data_GEOM)
                epoch_losses.append([l.numpy() for l in batch_loss_values])

            avg_losses = np.mean(epoch_losses, axis=0)
            total_loss, loss_a, loss_bc, loss_m, loss_u, loss_v, loss_pde_a, loss_geom = avg_losses

            history_loss_a.append(loss_a)
            history_loss_f_uv.append(loss_u + loss_v)
            history_loss_f_ma.append(loss_m + loss_pde_a)
            history_loss_geom.append(loss_geom)
            
            if epoch % checkpoint_interval == 0:
                current_time = time.time()
                time_for_epoch = current_time - start_checkpoint_time
                start_checkpoint_time = current_time
                log_msg = f"Epoch: {epoch}/{epochs} - Time: {time_for_epoch:.2f}s - Loss: {total_loss:.4e}"
                log_msg += f" | a: {loss_a:.4e}, BC: {loss_bc:.4e}, m: {loss_m:.4e}"
                log_msg += f", u: {loss_u:.4e}, v: {loss_v:.4e}, pde_a: {loss_pde_a:.4e}"
                log_msg += f" geom: {loss_geom:.4e}"
                logger.info(log_msg)
            
            # Saves weights every 'checkpoint_interval' epochs, overwriting the previous file.
            if epoch % checkpoint_interval == 0 and total_loss < current_best_total_loss:
                logger.info(f"Saving checkpoint at epoch {epoch} with loss {total_loss:.4e}")
                # Remove previous checkpoint file to ensure only one exists
                for f in glob.glob(os.path.join(dirname, "*_weights.h5")):
                    os.remove(f)
                # Save new checkpoint with loss in the filename
                safe_loss = f"{total_loss:.4e}".replace("+", "").replace("-", "m")
                weight_filename = f"loss_{safe_loss}.weights.h5"
                pinn.nn.save_weights(os.path.join(dirname, weight_filename))
                current_best_total_loss = total_loss

    total_training_time = time.time() - start_total
    logger.info(f"\nTotal training time: {total_training_time:.3f}s")
    
    logger.info("\n" + "="*50)
    logger.info("PERFORMING FINAL EVALUATION AND REPORTING")
    logger.info("="*50)

    list_of_files = glob.glob(os.path.join(dirname, '*.weights.h5'))
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        logger.info(f"Loading best model weights from: {os.path.basename(latest_file)}\n")
        pinn.nn.load_weights(latest_file)
    else:
        logger.info("No checkpoint file found. Evaluating with final weights from training.\n")

    logger.info("Calculating final loss...")
    final_evaluation_losses = []
    # Using batching parameters from the last training phase
    for b in range(num_batches):
        batch_dict = {}
        for key, df in training_data.items(): # Using original, non-shuffled data
            start_idx = b * prop_batch_sizes[key]
            end_idx = (b + 1) * prop_batch_sizes[key]
            batch_dict[key] = df.iloc[start_idx:end_idx]

        if all(batch.empty for batch in batch_dict.values()):
            continue
            
        data_A = to_tensor_tuple(batch_dict['A'], batch_dict['A'].columns)
        data_PDE = to_tensor_tuple(batch_dict['PDE'], ['x_PDE', 'y_PDE', 't_PDE'])
        data_N = to_tensor_tuple(batch_dict['N'], batch_dict['N'].columns)
        data_EW = to_tensor_tuple(batch_dict['EW'], ['x_E', 'y_E', 't_EW', 'x_W', 'y_W'])
        data_NSEW = to_tensor_tuple(batch_dict['NSEW'], batch_dict['NSEW'].columns)
        data_GEOM = to_tensor_tuple(batch_dict['GEOM'], batch_dict['GEOM'].columns)
        
        # Compute loss for the batch without training
        batch_losses = pinn.compute_loss(data_A, data_PDE, data_N, data_EW, data_NSEW, data_GEOM)
        final_evaluation_losses.append([l.numpy() for l in batch_losses])

    # Calculate the mean loss across all batches
    avg_final_losses = np.mean(final_evaluation_losses, axis=0)
    _, loss_a, loss_bc, loss_m, loss_u, loss_v, loss_pde_a, loss_geom = avg_final_losses

    logger.info("--- Final Loss Breakdown ---")
    logger.info(f"MSE_alpha (volume fraction): {loss_a:.4e}")
    logger.info(f"MSE_BC                     : {loss_bc:.4e}")
    logger.info(f"MSE_f,m                    : {loss_m:.4e}")
    logger.info(f"MSE_f,u                    : {loss_u:.4e}")
    logger.info(f"MSE_f,v                    : {loss_v:.4e}")
    logger.info(f"MSE_f,a                    : {loss_pde_a:.4e}")
    logger.info(f"MSE_GEOM                    : {loss_geom:.4e}")
    logger.info("----------------------------\n")

    # --- Plotting and Saving History (No changes needed here) ---
    logger.info("Generating and saving loss history plots...")
    epochs_range = range(1, len(history_loss_a) + 1)
    
    # Plot 1: MSE_alpha
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history_loss_a)
    plt.title(f'MSE of Volume Fraction (alpha) vs. Epochs ({adaptive_mode} - {activation_choice})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(dirname, 'loss_history_alpha.png'))
    
    # Plot 2: MSE_f,u + MSE_f,v
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history_loss_f_uv)
    plt.title(f'MSE of Momentum (u,v) vs. Epochs ({adaptive_mode} - {activation_choice})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (f_u + f_v)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(dirname, 'loss_history_momentum_uv.png'))
    
    # Plot 3: MSE_f,m + MSE_f,a
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history_loss_f_ma)
    plt.title(f'MSE of Conservation (m,a) vs. Epochs ({adaptive_mode} - {activation_choice})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (f_m + f_a)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(dirname, 'loss_history_conservation_ma.png'))

    # Plot 4: MSE GEOM
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history_loss_geom)
    plt.title(f'MSE of Geometric Loss Grad vs Int. Norm. vs. Epochs ({adaptive_mode} - {activation_choice})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss Geom')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(dirname, 'loss_history_geom.png'))
    
    plt.close('all') # Close all figures to free memory
    logger.info("Plots saved successfully.")

    # Save history to a CSV file with a descriptive name
    history_filename = f"loss_history_{adaptive_mode}_{activation_choice}.csv"
    history_filepath = os.path.join(dirname, history_filename)
    
    history_df = pd.DataFrame({
        'epoch': epochs_range,
        'MSE_alpha': history_loss_a,
        'MSE_f_uv': history_loss_f_uv,
        'MSE_f_ma': history_loss_f_ma,
        'MSE_geom': history_loss_geom
    })
    
    history_df.to_csv(history_filepath, index=False)
    logger.info(f"Loss history data saved to: {history_filepath}")
    
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

if __name__ == "__main__":
    main()