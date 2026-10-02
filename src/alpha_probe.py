"""Diagnostic: can the network fit the volume-fraction field AT ALL?

Trains only the alpha output against the alpha data -- no PDE, no boundary
conditions. This separates two very different failure modes when loss_a plateaus
at the constant-predictor baseline during real training:

  * drops well below baseline  -> network/points/labels are fine, and the PDE
                                  terms are suppressing the data fit
                                  (a loss-balancing problem)
  * stays at baseline          -> the data itself is the problem: labels,
                                  or L_ref/T_ref putting inputs in a bad range

Run it the same way as training, so it gets the GPU:
    LD_LIBRARY_PATH=$(python -c "import nvidia,glob,os;print(':'.join(sorted(glob.glob(os.path.join(os.path.dirname(nvidia.__file__),'*','lib')))))") python -u alpha_probe.py
"""
import sys
sys.path.append("../utilities")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import time
import numpy as np
import tensorflow as tf
from generate_points_3d import get_training_data_3d
from utilities import NNCreator

STEPS = 3000
BATCH = 4000
LR = 1e-4

np.random.seed(1234)
tf.random.set_seed(1234)

gpus = tf.config.list_physical_devices('GPU')
print(f"[{time.strftime('%H:%M:%S')}] devices: {[g.name for g in gpus] or 'CPU ONLY (will be ~50x slower)'}",
      flush=True)

t0 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] generating points ...", flush=True)
frames, gauge, scales = get_training_data_3d((500, 400), (400, 2000, 3000))
A = frames['A']
print(f"[{time.strftime('%H:%M:%S')}] done in {time.time()-t0:.1f}s -- {len(A)} alpha points", flush=True)

X = tf.constant(A[['x_A', 'y_A', 'z_A', 't_A']].to_numpy(), tf.float32)
Y = tf.constant(A[['a_A']].to_numpy(), tf.float32)
baseline = float(np.var(A.a_A.to_numpy()))
print(f"[{time.strftime('%H:%M:%S')}] mean a = {float(A.a_A.mean()):.4f}, "
      f"constant-predictor baseline = {baseline:.4f}", flush=True)

activation = {i: ['tanh', None, 10] for i in range(1, 9)}
outputs = list(zip(["output_u", "output_v", "output_w", "output_p", "output_a"],
                   [None, None, None, "exponential", "sigmoid"]))
net = NNCreator(tf.float32).get_model_dnn(4, [400] * 8, outputs, activation, False,
                                          periodic_lengths=scales['period'])
if os.path.exists('initial_weights_3d.h5'):
    net.load_weights('initial_weights_3d.h5')
    print(f"[{time.strftime('%H:%M:%S')}] loaded initial_weights_3d.h5", flush=True)

opt = tf.keras.optimizers.Adam(LR)
opt.build(net.trainable_variables)


@tf.function
def step(xb, yb):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(yb - net(xb)[4]))
    opt.apply_gradients(zip(tape.gradient(loss, net.trainable_variables),
                            net.trainable_variables))
    return loss


print(f"[{time.strftime('%H:%M:%S')}] tracing first step (a few seconds) ...", flush=True)
n = len(A)
t0 = time.time()
for i in range(STEPS + 1):
    idx = np.random.randint(0, n, BATCH)
    loss = step(tf.gather(X, idx), tf.gather(Y, idx))
    if i == 0:
        print(f"[{time.strftime('%H:%M:%S')}] first step done in {time.time()-t0:.1f}s", flush=True)
    if i % 100 == 0:
        L = float(loss)
        bar = '#' * int(40 * min(1.0, L / baseline))
        print(f"  step {i:5d}  loss_a {L:.4f}  {100*L/baseline:5.1f}% of baseline  |{bar:<40}|",
              flush=True)

L = float(loss)
print(f"\nfinal loss_a = {L:.4f}  ({100*L/baseline:.1f}% of baseline {baseline:.4f})")
print("VERDICT:", "data fits fine -> the PDE terms are suppressing it (balancing problem)"
      if L < 0.5 * baseline else
      "STILL AT BASELINE -> the alpha data/labels/scaling are the problem")
