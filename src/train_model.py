# TwoPhasePinn in TensorFlow 2.x
import os, sys, time, math, glob, shutil, logging
from datetime import datetime
sys.path.append("../utilities")
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
from tensorflow.keras import backend as K
from generate_points import *
from utilities import *
from logger import TrainingLogger
np.random.seed(1234)
tf.random.set_seed(1234)

class TwoPhasePinn:

    def __init__(self, dtype, hidden_layers, activation_functions, adaptive_activation_coeff, adaptive_activation_n, 
        adaptive_activation_init, use_ad_act, loss_weights_A, loss_weights_PDE, mu, sigma, g, rho, u_ref, L_ref, 
        checkpoint_interval, epochs, batch_sizes, learning_rates):

        self.dtype = dtype
        self.dirname, logpath = self.make_output_dir()
        self.logger = TrainingLogger(logpath)

        self.sanity_check_activation_functions(activation_functions, adaptive_activation_coeff, adaptive_activation_n, adaptive_activation_init, hidden_layers)
        self.ad_act_coeff = {}
        if use_ad_act:
            for key in adaptive_activation_coeff:
                initial_value = adaptive_activation_init[key]
                self.ad_act_coeff[key] = tf.Variable(initial_value, name=key, dtype=dtype)

        activation_functions_dict = self.get_activation_function_dict(activation_functions, adaptive_activation_coeff, adaptive_activation_n, hidden_layers, use_ad_act)

        self.mu1, self.mu2 = mu
        self.sigma, self.g = sigma, g
        self.rho1, self.rho2 = rho
        self.U_ref, self.L_ref = u_ref, L_ref

        self.epoch_loss_checkpoints = 1e10
        self.checkpoint_interval = checkpoint_interval
        self.mean_epoch_time = 0

        self.learning_rates = learning_rates
        self.epochs = epochs
        self.batch_sizes = batch_sizes

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rates[0])

        self.logger.log("Building Computational Graph")

        # Placeholders replaced with just column names
        self.placeholders = {
            "A":    ["x_A", "y_A", "t_A", "a_A"],
            "PDE":  ["x_PDE", "y_PDE", "t_PDE", "f_PDE"],
            "N":    ["x_N", "y_N", "t_N", "p_N"],
            "EW":   ["x_E", "y_E", "x_W", "y_W", "t_EW"],
            "NSEW": ["x_NSEW", "y_NSEW", "t_NSEW", "u_NSEW", "v_NSEW"]
        }

        outputs = ["output_u", "output_v", "output_p", "output_a"]
        activations_output = [None, None, "exponential", "sigmoid"]
        output_layer = list(zip(outputs, activations_output))

        nn = NNCreator(dtype)
        self.model = nn.get_model_dnn(3, hidden_layers, output_layer, activation_functions_dict, use_ad_act)

        self.loss_list = ["l", "a", "NSEW", "m", "u", "v", "PDE_a"]
        self.loss_history = {loss: [] for loss in self.loss_list}
        self.ad_act_coeff_history = {key: [] for key in self.ad_act_coeff}
        self.epoch_loss = {loss: 0.0 for loss in self.loss_list}


        self.set_variables()
        self.model.save_weights(
            os.path.join(self.dirname, "Weights_loss_%.4e.weights.h5" % self.epoch_loss_checkpoints)
        )



    def check_matching_keys(self, data_sets):
        for key1, key2 in zip(data_sets, self.placeholders):
            assert key1 == key2, "Data set key %s does not match placeholder key %s" % (key1, key2)

    def print_point_distribution(self, data_sets):
        no_points = 0
        for key in data_sets:
            no_points += data_sets[key].shape[0]
            self.logger.log("Training data %10s shape: %s" %(key, data_sets[key].shape))
        self.logger.log("Total number of points %d" % no_points)

    def shuffle_data_and_reset_epoch_losses(self, data_sets):
        for key in data_sets:
            length = len(data_sets[key])
            shuffled_indices = np.random.choice(length, length, replace=False) 
            data_sets[key] = pd.DataFrame(data=data_sets[key].to_numpy()[shuffled_indices,:], columns=data_sets[key].columns)
        for key in self.epoch_loss:
            self.epoch_loss[key] = 0
        return data_sets

    def get_batches(self, data, b, batch_sizes):
        batches = dict.fromkeys(data.keys(), 0)
        for key in data:
            batches[key] = data[key][b*batch_sizes[key]:(b+1)*batch_sizes[key]]
        return batches

    def get_feed_dict(self, batches, counter):
        tf_dict = {'learning_rate': self.learning_rates[counter]}
        # loop over the groups
        for key in self.placeholders:
            # collect the tensors for this group
            tensors = []
            for col_name in self.placeholders[key]:
                col_data = batches[key][col_name].to_numpy()
                col_data = np.atleast_2d(col_data).T if col_data.ndim == 1 else col_data
                tensors.append(tf.convert_to_tensor(col_data, dtype=self.dtype))
            tf_dict[key] = tensors
        return tf_dict
   

    def get_batch_sizes(self, counter, data_sets):
        number_of_samples = sum([len(data_sets[key]) for key in data_sets])
        batch_sizes_datasets = dict.fromkeys(data_sets.keys(), 0)
        if self.batch_sizes[counter] >= number_of_samples:
            number_of_batches = 1
            for key in data_sets:
                batch_sizes_datasets[key] = len(data_sets[key])
            self.logger.log("Batch size is larger equal the amount of training samples, thus going full batch mode")
            msg = (
                f"Total batch size: {number_of_samples} - Batch sizes: {batch_sizes_datasets} - learning rate: {self.learning_rates[counter]}\n"
            )
            self.logger.log()
        else:
            number_of_batches = math.ceil(number_of_samples/self.batch_sizes[counter])
            batch_percentages = dict.fromkeys(data_sets.keys(), 0)
            print_batches = dict.fromkeys(data_sets.keys(), "")
            for key in data_sets:
                batch_percentages[key] = len(data_sets[key])/number_of_samples
                batch_sizes_datasets[key] = math.ceil(self.batch_sizes[counter]*batch_percentages[key])
                print_batches[key] = "%d/%d" % (batch_sizes_datasets[key], 0 if batch_sizes_datasets[key] == 0 else len(data_sets[key])%batch_sizes_datasets[key])
            total_batch_size = sum([batch_sizes_datasets[key] for key in batch_sizes_datasets])
            self.logger.log(f"\nTotal batch size: {total_batch_size} - number of batches: {number_of_batches} - Batch sizes: {print_batches} - learning rate: {self.learning_rates[counter]}")
            for key in data_sets:
                if len(data_sets[key]) == 0:
                    continue
                assert (number_of_batches - 1) * batch_sizes_datasets[key] < len(data_sets[key]), "The specified batch size of %d will lead to empty batches with the present batch ratio, increase the batch size!" % (self.batch_sizes[counter])
        return batch_sizes_datasets, number_of_batches


    def print_info(self, current_epoch, epochs, time_for_epoch):
        if current_epoch == 1:  # skip first epoch for avg
            self.mean_epoch_time = 0
        else:
            self.mean_epoch_time = (
                self.mean_epoch_time * (current_epoch - 2) / (current_epoch - 1)
                + time_for_epoch / (current_epoch - 1)
            )

        string = [
            f"Epoch: {current_epoch:5d}/{epochs} - {time_for_epoch*1e3:7.2f}ms - avg: {self.mean_epoch_time*1e3:7.2f}ms"
        ]

        def to_scalar(val):
            if isinstance(val, tf.Tensor):
                val = val.numpy()
            if isinstance(val, np.ndarray):
                if val.size == 1:
                    return float(val.item())
                else:
                    return float(val.mean())
            return float(val)

        # Append losses
        for key, value in self.epoch_loss.items():
            string.append(f" - {key}: {to_scalar(value):.6e}")

        # Append adaptive activation coefficients
        for key, act_coeff in self.ad_act_coeff.items():
            string.append(f" - {key}: {to_scalar(act_coeff):.6e}")

        self.logger.log(" | ".join(string))




    def compute_gradients(self, x, y, t):
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y, t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y, t])
                u, v, p, a = self.model(tf.concat([x, y, t], axis=1))
            # first-order derivatives
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

        # second-order derivatives
        u_xx = tape2.gradient(u_x, x)
        u_yy = tape2.gradient(u_y, y)

        v_xx = tape2.gradient(v_x, x)
        v_yy = tape2.gradient(v_y, y)

        a_xx = tape2.gradient(a_x, x)
        a_yy = tape2.gradient(a_y, y)
        a_xy = tape2.gradient(a_x, y)

        del tape1
        del tape2

        return (
            [u, u_x, u_y, u_t, u_xx, u_yy],
            [v, v_x, v_y, v_t, v_xx, v_yy],
            [p, p_x, p_y],
            [a, a_x, a_y, a_t, a_xx, a_yy, a_xy]
        )

    def save_model_checkpoint(self, loss, epoch, counter):
        '''
        Save model and metadata when a checkpoint is reached.
        
        Files written to self.dirname:
          - full model (.keras)
          - architecture (.json)
          - weights (.weights.h5)
          - loss history & adaptive activation coeffs (.mat)
        '''

        if loss < self.epoch_loss_checkpoints and epoch % self.checkpoint_interval == 0:
            for file in glob.glob(os.path.join(self.dirname, "*")):
                if file.endswith((".weights.h5", ".mat", ".json", ".keras")):
                    os.remove(file)

            # Save full model (.keras)
            keras_path = os.path.join(self.dirname, f"loss_{loss:.4e}_model.keras")
            self.model.save(keras_path)  # saves architecture + weights + optimizer state
            self.logger.log(f"✅ Saved full model to {keras_path}")

            # Save just architecture (.json)
            json_path = os.path.join(self.dirname, f"loss_{loss:.4e}_architecture.json")
            with open(json_path, "w") as json_file:
                json_file.write(self.model.to_json())
            self.logger.log(f"✅ Saved model architecture to {json_path}")

            # Save just weights (.h5)
            weights_path = os.path.join(self.dirname, f"loss_{loss:.4e}_weights.h5")
            self.model.save_weights(weights_path)
            self.logger.log(f"✅ Saved model weights to {weights_path}")

            # Save metadata (.mat)
            ad_act_coeff_values = {k: v.numpy() for k, v in self.ad_act_coeff.items()}
            data = dict(
                loss_history=self.loss_history,
                ad_act_coeff_history=self.ad_act_coeff_history,
                ad_act_coeff=ad_act_coeff_values,
                epoch=epoch,
                learning_rate=self.learning_rates[counter]
            )
            mat_path = os.path.join(self.dirname, f"loss_{loss:.4e}_variables.mat")
            scipy.io.savemat(mat_path, data)
            self.logger.log(f"✅ Saved training metadata to {mat_path}")

            # Update the best loss tracker
            self.epoch_loss_checkpoints = loss


    def PDE_caller(self, x, y, t):
        u_gradients, v_gradients, p_gradients, a_gradients = self.compute_gradients(x, y, t)
        u, u_x, u_y, u_t, u_xx, u_yy = u_gradients[:]
        v, v_x, v_y, v_t, v_xx, v_yy = v_gradients[:]
        p, p_x, p_y = p_gradients[:]
        a, a_x, a_y, a_t, a_xx, a_yy, a_xy = a_gradients[:]

        mu = self.mu2 + (self.mu1 - self.mu2) * a
        mu_x = (self.mu1 - self.mu2) * a_x
        mu_y = (self.mu1 - self.mu2) * a_y
        rho = self.rho2 + (self.rho1 - self.rho2) * a

        abs_interface_grad = tf.sqrt(tf.square(a_x) + tf.square(a_y) + np.finfo(float).eps)

        curvature = - ( (a_xx + a_yy)/abs_interface_grad - (a_x**2*a_xx + a_y**2*a_yy + 2*a_x*a_y*a_xy)/tf.pow(abs_interface_grad, 3) )

        rho_ref = self.rho2

        one_Re = mu/(rho_ref*self.U_ref*self.L_ref)
        one_Re_x = mu_x/(rho_ref*self.U_ref*self.L_ref)
        one_Re_y = mu_y/(rho_ref*self.U_ref*self.L_ref)
        one_We = self.sigma/(rho_ref*self.U_ref**2*self.L_ref)
        one_Fr = self.g*self.L_ref/self.U_ref**2 

        PDE_m = u_x + v_y
        PDE_a = a_t + u*a_x + v*a_y
        PDE_u = (u_t + u*u_x + v*u_y)*rho/rho_ref + p_x - one_We*curvature*a_x - one_Re*(u_xx + u_yy) - 2.0*one_Re_x*u_x - one_Re_y*(u_y + v_x) 
        PDE_v = (v_t + u*v_x + v*v_y)*rho/rho_ref + p_y - one_We*curvature*a_y - one_Re*(v_xx + v_yy) - rho/rho_ref*one_Fr - 2.0*one_Re_y*v_y - one_Re_x*(u_y + v_x) 

        return PDE_m, PDE_u, PDE_v, PDE_a

    def compute_loss(self, tf_dict):
        # Learning rate (if needed)
        lr = tf_dict['learning_rate']

        # Unpack batches
        A, PDE, N, EW, NSEW = [tf_dict[k] for k in ["A", "PDE", "N", "EW", "NSEW"]]

        x_A, y_A, t_A, a_A = A
        x_N, y_N, t_N, p_N = N
        x_E, y_E, x_W, y_W, t_EW = EW
        x_NSEW, y_NSEW, t_NSEW, u_NSEW, v_NSEW = NSEW
        x_PDE, y_PDE, t_PDE, f_PDE = PDE

        # Evaluate model outputs at different input points
        # A points
        out_u_A, out_v_A, out_p_A, out_a_A = self.model(tf.concat([x_A, y_A, t_A], axis=1))
        loss_a_A = tf.reduce_mean(tf.square(a_A - out_a_A))

        # NSEW points
        out_u_NSEW, out_v_NSEW, out_p_NSEW, out_a_NSEW = self.model(tf.concat([x_NSEW, y_NSEW, t_NSEW], axis=1))
        loss_u_NSEW = tf.reduce_mean(tf.square(u_NSEW - out_u_NSEW))
        loss_v_NSEW = tf.reduce_mean(tf.square(v_NSEW - out_v_NSEW))
        loss_NSEW = loss_u_NSEW + loss_v_NSEW

        # N points (pressure)
        out_u_N, out_v_N, out_p_N, out_a_N = self.model(tf.concat([x_N, y_N, t_N], axis=1))
        loss_p_N = tf.reduce_mean(tf.square(p_N - out_p_N))

        # EW points
        out_u_E, out_v_E, out_p_E, out_a_E = self.model(tf.concat([x_E, y_E, t_EW], axis=1))
        out_u_W, out_v_W, out_p_W, out_a_W = self.model(tf.concat([x_W, y_W, t_EW], axis=1))
        loss_EW_u = tf.reduce_mean(tf.square(out_u_E - out_u_W))
        loss_EW_v = tf.reduce_mean(tf.square(out_v_E - out_v_W))
        loss_EW_p = tf.reduce_mean(tf.square(out_p_E - out_p_W))
        loss_EW = loss_EW_u + loss_EW_v + loss_EW_p

        loss_NSEW += loss_p_N + loss_EW

        # PDE points
        PDE_tensors = self.PDE_caller(x_PDE, y_PDE, t_PDE)
        loss_PDE_components = [tf.reduce_mean(tf.square(f_PDE - t)) for t in PDE_tensors]

        loss_PDE = tf.tensordot(loss_PDE_components, tf.constant([1.0, 10.0, 10.0, 1.0]), axes=1)

        loss_complete = loss_a_A + loss_NSEW + loss_PDE

        loss_components = [loss_complete, loss_a_A, loss_NSEW] + list(PDE_tensors)

        return loss_complete, loss_components

    def make_output_dir(self):
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        dirname = os.path.abspath(os.path.join("checkpoints", datetime.now().strftime("%b-%d-%Y_%H-%M-%S")))
        os.mkdir(dirname)
        script_filename = os.path.basename(__file__)
        shutil.copyfile(__file__, os.path.join(dirname, script_filename))
        shutil.copyfile("generate_points.py", os.path.join(dirname, "generate_points.py"))
        
        logpath = os.path.join(dirname, "output.log")
        return dirname, logpath

    def get_activation_function_dict(self, activation_functions, adaptive_activation_coeff, adaptive_activation_n, hidden_layers, use_ad_act):
        activation_functions_dict = dict((key, [0, 0, 0]) for key in range(1, len(hidden_layers) + 1))
        for layer_no in activation_functions_dict:
            activation_functions_dict[layer_no][2] = adaptive_activation_n[layer_no-1]
            for func_name, layers in activation_functions.items():
                if layer_no in layers:
                    activation_functions_dict[layer_no][0] = func_name
            if use_ad_act:                                                  # if use_ad_act is False, self.ad_act_coeff is empty!
                for coeff_name, layers in adaptive_activation_coeff.items():
                    if layer_no in layers:
                        activation_functions_dict[layer_no][1] = self.ad_act_coeff[coeff_name]
        return activation_functions_dict

    def sanity_check_activation_functions(self, activation_functions, adaptive_activations, adaptive_activation_n, adaptive_activation_init, hidden_layers):
        no_layers = len(hidden_layers)
        check = 0
        for key, value in list(adaptive_activations.items()):                  
            check += sum(value)
        assert no_layers*(no_layers+1)/2 == check, "Not every layer has been assigned with an adaptive activation coefficient unambiguously"
        check = 0
        for key, value in list(activation_functions.items()):                  
            check += sum(value)
        assert no_layers*(no_layers+1)/2 == check, "Not every layer has been assigned with an activation function unambiguously"
        assert no_layers == len(adaptive_activation_n), "Not every layer has an adaptive activation precoefficient"
        assert adaptive_activation_init.keys() == adaptive_activations.keys(), "Not every adaptive activation coefficient has been assigned an initial value"


    def get_logger(self, logpath):
        logger_name = f"TwoPhasePinn-{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        if logger.hasHandlers():
            logger.handlers.clear()

        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)        
        sh.setFormatter(logging.Formatter('%(message)s'))

        fh = logging.FileHandler(logpath)
        fh.setLevel(logging.DEBUG)

        logger.addHandler(sh)
        logger.addHandler(fh)
        return logger

    @tf.function(jit_compile=True)
    def train_step(self, tf_dict):
        with tf.GradientTape() as tape:
            loss, batch_losses = self.compute_loss(tf_dict)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, batch_losses


    def build_tf_dataset(self, df, columns, batch_size):
        data = [tf.convert_to_tensor(df[c].values.reshape(-1,1), dtype=self.dtype) for c in columns]
        dataset = tf.data.Dataset.from_tensor_slices(tuple(data))
        dataset = dataset.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

        dataset = dataset.cache()

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def zip_datasets(self, datasets):
        keys = list(datasets.keys())
        zipped = tf.data.Dataset.zip(tuple(datasets[k] for k in keys))
        return zipped

    def build_epoch_datasets(self, data_sets, batch_sizes):
        datasets = {}
        for key in self.placeholders:
            datasets[key] = self.build_tf_dataset(
                data_sets[key], 
                self.placeholders[key], 
                batch_sizes[key]
            )
        return datasets

    @tf.function
    def train_batch(self, batch, counter):
        keys = list(self.placeholders.keys())
        tf_dict = {'learning_rate': self.learning_rates[counter]}
        for i, key in enumerate(keys):
            tf_dict[key] = batch[i]
        loss, batch_losses = self.train_step(tf_dict)
        return loss, batch_losses

    def preprocess_data(self, data_sets):
        """Convert pandas DataFrames into dict of lists of tensors, one per column."""
        tensor_data_sets = {}
        for key in self.placeholders:
            tensor_data_sets[key] = []
            for col in self.placeholders[key]:
                col_data = data_sets[key][col].values.reshape(-1,1)
                tensor_data_sets[key].append(tf.convert_to_tensor(col_data, dtype=self.dtype))
        return tensor_data_sets


    def build_datasets(self, tensor_data_sets, batch_sizes):
        """Build tf.data.Dataset objects, shuffled & batched, ready for training."""
        datasets = {}
        for key in self.placeholders:
            data = tensor_data_sets[key]
            dataset = tf.data.Dataset.from_tensor_slices(tuple(data))
            dataset = dataset.shuffle(buffer_size=len(data[0]), reshuffle_each_iteration=True)
            dataset = dataset.batch(batch_sizes[key])
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            datasets[key] = dataset
        return datasets


    def zip_datasets(self, datasets):
        """Zip the individual datasets into a single dataset yielding batches of all."""
        keys = list(self.placeholders.keys())
        zipped = tf.data.Dataset.zip(tuple(datasets[k] for k in keys))
        return zipped


    def train(self, data_sets):
        self.check_matching_keys(data_sets)
        self.print_point_distribution(data_sets)
        msg = (
            f"\n[bold green]EPOCHS:[/bold green] {self.epochs}\n"
            f"[bold blue]BATCH SIZES:[/bold blue] {self.batch_sizes}\n"
            f"[bold magenta]LEARNING RATES:[/bold magenta] {self.learning_rates}"
        )
        self.logger.log(msg)

        start_total = time.time()

        # Preprocess data once
        tensor_data_sets = self.preprocess_data(data_sets)
        for counter, epoch_value in enumerate(self.epochs):
            self.current_learning_rate = self.learning_rates[counter]
            self.optimizer.learning_rate.assign(self.learning_rates[counter])
            batch_sizes, _ = self.get_batch_sizes(counter, data_sets)

            # Build datasets once for this phase
            datasets = self.build_datasets(tensor_data_sets, batch_sizes)
            zipped_dataset = self.zip_datasets(datasets)

            # Warm up @tf.function to avoid first-step lag
            for batch in zipped_dataset.take(1):
                _ = self.train_batch(batch, counter)

            for e in range(1, epoch_value + 1):
                start_epoch = time.time()
                self.epoch_loss = {k: 0.0 for k in self.loss_list}

                num_batches = 0
                for batch in zipped_dataset:
                    loss, batch_losses = self.train_batch(batch, counter)
                    for i, key in enumerate(self.loss_list):
                        self.epoch_loss[key] += batch_losses[i]  # tensor
                    num_batches += 1

                # average over batches
                for key in self.loss_list:
                    self.epoch_loss[key] /= num_batches

                # checkpoint & print
                self.save_model_checkpoint(loss.numpy(), e, counter)
                self.print_info(e, self.epochs[counter], time.time() - start_epoch)

        self.logger.log("\nTotal training time: %.3fs" % (time.time() - start_total))

    def set_variables(self):
        for file in glob.glob("*loss*"):
            if file.endswith("h5"):
                self.model.load_weights(file)
                self.logger.log("Loading weights from file", file)
            if file.endswith("mat"):
                matfile = scipy.io.loadmat(file, squeeze_me=True)
                self.logger.log("Setting adaptive activation coefficients")
                ad_act_coeff = matfile["ad_act_coeff"]
                for key in self.ad_act_coeff:
                    self.ad_act_coeff[key].assign(float(ad_act_coeff[key]))

    def print(self, *args):
        for word in args:
            if len(args) == 1:
                self.logger.info(word)
            else:
                self.logger.info(" ".join(str(a) for a in args))


def compute_batch_size(training_data, number_of_batches):
    number_of_samples = sum(len(training_data[key]) for key in training_data)
    return math.ceil(number_of_samples/number_of_batches)


if __name__ == "__main__":
    dtype = tf.float32
    NOP_a = (500, 400)
    NOP_PDE = (400, 2000, 3000)
    NOP_north = (20, 20)
    NOP_south = (20, 20)
    NOP_east = (20, 20)
    NOP_west = (20, 20)

    training_data = get_training_data(NOP_a, NOP_PDE, NOP_north, NOP_south, NOP_east, NOP_west)

    no_layers = 8
    hidden_layers = [400]*no_layers
    activation_functions = dict(tanh=range(1,no_layers+1))

    adaptive_activation_coeff = {"aac_1": range(1,no_layers+1)}
    adaptive_activation_init = {"aac_1": 0.1}
    adaptive_activation_n = [10]*no_layers
    use_ad_act = False

    mu = [1.0, 10.0]
    sigma = 24.5
    g = -0.98
    rho = [100, 1000]
    u_ref = 1.0
    L_ref = 0.25

    loss_weights_A = [1.0]
    loss_weights_PDE = [1.0, 10.0, 10.0, 1.0]
    epochs = [100]*5
    number_of_batches = 10
    batch_sizes = [compute_batch_size(training_data, number_of_batches)]*5
    learning_rates = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    checkpoint_interval = 2500
    tf.config.optimizer.set_jit(True)

    PINN = TwoPhasePinn(dtype, hidden_layers, activation_functions, adaptive_activation_coeff, adaptive_activation_n, 
        adaptive_activation_init, use_ad_act, loss_weights_A, loss_weights_PDE, mu, sigma, g, rho, u_ref, L_ref, 
        checkpoint_interval, epochs, batch_sizes, learning_rates)


    PINN.train(training_data)
