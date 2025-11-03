import os
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import json
import mat4py
import torch
import torch.nn as nn


def reshape_test_data(x, y, t):
    """Function reshaping the test data to feed to neural network for prediction"""
    Y, T, X = np.meshgrid(y, t, x)
    X = X.reshape(X.size, 1)
    Y = Y.reshape(Y.size, 1)
    T = T.reshape(T.size, 1)
    test_data = np.concatenate((X, Y, T), axis=1)
    return test_data


def reshape_prediction(x, y, t, u):
    """Function reshaping the predictions into 2D arrays"""
    return u.reshape(len(t), len(y), len(x), order="C")


def load_nn_model(path, plot_loss_history=True, device='cuda'):
    """
    Loads the PyTorch model from files located in path.
    
    Required files:
    1) .json containing the architecture
    2) .pth containing the weights
    3) .mat containing loss history, adaptive activation history and optimizer state.
    
    Args:
        path: absolute path to where the required files are located
        plot_loss_history: bool indicating whether to plot the loss history
        device: device to load the model on ('cuda' or 'cpu')
    """
    matfile = None
    architecture = None
    weights = None
    
    for file in os.listdir(path):
        if file.endswith("mat") and "weights" not in file:
            matfile = mat4py.loadmat(os.path.join(path, file))
        if file.endswith("json"):
            with open(os.path.join(path, file)) as json_data:
                architecture = json.load(json_data)
        if file.endswith("pth"):
            weights = os.path.join(path, file)
    
    if weights is None:
        raise FileNotFoundError("No .pth weight file found in the specified path")
    
    print(f"\nLoading nn model: {weights}")
    
    # LOAD LOSS HISTORY
    if matfile is not None and "loss_history" in matfile:
        loss_history = matfile["loss_history"]
        
        if plot_loss_history:
            fig, ax = plt.subplots()
            fig.set_size_inches([15, 8])
            for key in loss_history:
                print(f"Final loss {key}: {loss_history[key][-1]:.4e}")
                ax.semilogy(loss_history[key], label=key)
            ax.set_xlabel("epochs", fontsize=15)
            ax.set_ylabel("loss", fontsize=15)
            ax.tick_params(labelsize=15)
            ax.legend()
            plt.show()
    
    # For PyTorch, you need to reconstruct the model architecture first
    # This is a placeholder - you'll need to import your actual model class
    # and instantiate it based on the architecture JSON
    print("Note: Model architecture must be reconstructed separately in PyTorch")
    print(f"Load weights using: model.load_state_dict(torch.load('{weights}'))")
    
    return weights, architecture


def load_cfd(start_index, end_index, temporal_step_size, spatial_step_size):
    """
    Loads the CFD results and returns them in numpy arrays.
    The CFD results contain 151 time snapshots from t = 0.0 until t = 3.0 
    with a resolution of 512x256. The returned resolution may be coarsened 
    using the input arguments.
    
    Args:
        start_index: start index of returned time snapshots
        end_index: end index of returned time snapshots
        temporal_step_size: index temporal resolution
        spatial_step_size: index spatial resolution
    """
    # PATH TO CFD SOLUTION
    path = "../cfd_data/rising_bubble.h5"
    
    # OPEN AND ASSIGN TO NUMPY ARRAYS
    with h5py.File(path, "r") as data:
        X = np.array(data["X"])[::spatial_step_size]
        Y = np.array(data["Y"])[::spatial_step_size]
        time = np.array(data["time"])[start_index:end_index:temporal_step_size]
        levelset = np.array(data["levelset"])[start_index:end_index:temporal_step_size, 
                                              ::spatial_step_size, ::spatial_step_size]
        pressure = np.array(data["pressure"])[start_index:end_index:temporal_step_size, 
                                              ::spatial_step_size, ::spatial_step_size]
        velocityX = np.array(data["velocityX"])[start_index:end_index:temporal_step_size, 
                                                ::spatial_step_size, ::spatial_step_size]
        velocityY = np.array(data["velocityY"])[start_index:end_index:temporal_step_size, 
                                                ::spatial_step_size, ::spatial_step_size]
    
    print("Shape of CFD results:", pressure.shape)
    print("Loaded CFD time snapshots:\n", time)
    
    return pressure, velocityX, velocityY, levelset, X, Y, time


def update_contourf(i, xs, ys, data, axis, pcfsets, kwargs):
    """This function updates the contour plots"""
    list_of_collections = []
    
    for x, y, z, ax, pcfset, kw in zip(xs, ys, data, axis, pcfsets, kwargs):
        for tp in pcfset[0].collections:
            tp.remove()
        
        pcfset[0] = ax.contourf(x, y, z[i, :, :], **kw)
        list_of_collections += pcfset[0].collections
    
    return list_of_collections


def grid_contour_plots(data, nrows_ncols, titles, x, y, fontsize=15, labelsize=10):
    """
    Creates contour plots using an ImageGrid
    
    Args:
        data: list of numpy arrays containing values to plot
        nrows_ncols: amount of rows and columns of the ImageGrid
        titles: list of titles - needs to be same length as data
        x: numpy array containing spatial coordinates in x-direction
        y: numpy array containing spatial coordinates in y-direction
    """
    # CREATE FIGURE AND AXIS
    fig = plt.figure()
    grid = ImageGrid(fig, 111, direction="row", nrows_ncols=nrows_ncols, 
                    label_mode="1", axes_pad=0.8, share_all=False, 
                    cbar_mode="each", cbar_location="right", 
                    cbar_size="5%", cbar_pad=0.0)
    
    # CREATE ARGUMENTS DICT FOR CONTOURPLOTS
    minmax_list = []
    kwargs_list = []
    for d in data:
        minmax_list.append([np.min(d), np.max(d)])
        kwargs_list.append(dict(
            levels=np.linspace(minmax_list[-1][0], minmax_list[-1][1], 60),
            cmap="seismic", 
            vmin=minmax_list[-1][0], 
            vmax=minmax_list[-1][1]
        ))
    
    # CREATE PLOTS
    pcfsets = []
    for ax, z, kwargs, minmax, title in zip(grid, data, kwargs_list, minmax_list, titles):
        pcf = [ax.contourf(x, y, z[0, :, :], **kwargs)]
        pcfsets.append(pcf)
        
        cb = ax.cax.colorbar(pcf[0], ticks=np.linspace(minmax[0], minmax[1], 5))
        ax.cax.tick_params(labelsize=labelsize)
        ax.set_title(title, fontsize=fontsize, pad=10)
        ax.set_ylabel("y/R", labelpad=15, fontsize=fontsize, rotation="horizontal")
        ax.set_xlabel("x/R", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.set_aspect("equal")
    
    fig.set_size_inches(20, 10, True)
    
    return fig, grid, pcfsets, kwargs_list


def writeToJSONFile(path, fileName, data):
    """Write data to JSON file"""
    filePathNameWExt = path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)


class SineActivation(nn.Module):
    """Custom sine activation function"""
    def forward(self, x):
        return torch.sin(x)


class AdaptiveActivation(nn.Module):
    """Adaptive activation wrapper for PyTorch"""
    def __init__(self, activation_fn, coeff=1.0, n=1):
        super().__init__()
        self.activation_fn = activation_fn
        self.coeff = nn.Parameter(torch.tensor(coeff, dtype=torch.float32))
        self.n = n
    
    def forward(self, x):
        return self.activation_fn(self.coeff * self.n * x)


class DenseLayerWithActivation(nn.Module):
    """Dense layer with optional adaptive activation"""
    def __init__(self, in_features, out_features, activation_fn=None, 
                 use_adaptive=False, adaptive_coeff=1.0, adaptive_n=1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Xavier/Glorot normal initialization
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
        if activation_fn is not None:
            if use_adaptive:
                self.activation = AdaptiveActivation(activation_fn, adaptive_coeff, adaptive_n)
            else:
                self.activation = activation_fn
        else:
            self.activation = None
    
    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class NNCreator:
    """
    Class implementing a PyTorch DNN with optional adaptive activation coefficients
    """
    
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
    
    def get_model_dnn(self, input_dim, hidden_layers, output_layer, 
                     activation_functions_dict, use_ad_act):
        """
        Build a DNN model
        
        Args:
            input_dim: dimension of input (e.g., 3 for x, y, t)
            hidden_layers: list of hidden layer sizes
            output_layer: list of tuples (output_name, activation)
            activation_functions_dict: dict mapping layer index to [func_name, coeff, n]
            use_ad_act: whether to use adaptive activation
        
        Returns:
            PyTorch nn.Module
        """
        layers = []
        in_features = input_dim
        
        # Build hidden layers
        for i, nodes in enumerate(hidden_layers):
            layer_idx = i + 1
            activation_info = activation_functions_dict.get(layer_idx, [None, None, 1])
            activation_fn = self.get_activation_function(activation_info, use_ad_act)
            
            if use_ad_act and activation_info[0] is not None:
                coeff = activation_info[1] if activation_info[1] is not None else 1.0
                n = activation_info[2] if activation_info[2] else 1
                layer = DenseLayerWithActivation(in_features, nodes, activation_fn, 
                                                True, coeff, n)
            else:
                layer = DenseLayerWithActivation(in_features, nodes, activation_fn, 
                                                False, 1.0, 1)
            
            layers.append(layer)
            in_features = nodes
        
        # Build the model
        class DNN(nn.Module):
            def __init__(self, hidden_layers_list, output_specs, in_features):
                super().__init__()
                self.hidden_layers = nn.ModuleList(hidden_layers_list)
                
                # Create output layers
                self.output_layers = nn.ModuleDict()
                for output_name, activation in output_specs:
                    linear = nn.Linear(in_features, 1)
                    nn.init.xavier_normal_(linear.weight)
                    nn.init.zeros_(linear.bias)
                    self.output_layers[output_name] = linear
                
                self.output_specs = output_specs
            
            def forward(self, x):
                # Pass through hidden layers
                for layer in self.hidden_layers:
                    x = layer(x)
                
                # Generate outputs
                outputs = []
                for output_name, activation in self.output_specs:
                    out = self.output_layers[output_name](x)
                    
                    # Apply activation if specified
                    if activation == "exponential":
                        out = torch.exp(out)
                    elif activation == "sigmoid":
                        out = torch.sigmoid(out)
                    elif activation == "tanh":
                        out = torch.tanh(out)
                    elif activation == "sin" or activation == "sine":
                        out = torch.sin(out)
                    # None or other: no activation (linear)
                    
                    outputs.append(out)
                
                return tuple(outputs) if len(outputs) > 1 else outputs[0]
        
        model = DNN(layers, output_layer, in_features)
        return model
    
    def get_activation_function(self, name_coeff_n, use_ad_act):
        """
        Get activation function based on name
        
        Args:
            name_coeff_n: list [function_name, ad_act_coeff, n]
            use_ad_act: whether using adaptive activation
        
        Returns:
            activation function (nn.Module or None)
        """
        function_name = name_coeff_n[0]
        
        if function_name is None:
            return None
        
        # Map function names to PyTorch activations
        if function_name == "tanh":
            return nn.Tanh()
        elif function_name in ["sin", "sine"]:
            return SineActivation()
        elif function_name in ["logistic", "sigmoid"]:
            return nn.Sigmoid()
        elif function_name == "exponential":
            # Note: exponential activation is handled in forward pass
            # to avoid overflow issues
            return lambda x: torch.exp(torch.clamp(x, max=20))
        elif function_name == "relu":
            return nn.ReLU()
        elif function_name == "leaky_relu":
            return nn.LeakyReLU()
        elif function_name == "elu":
            return nn.ELU()
        else:
            print(f"Warning: Unknown activation function '{function_name}', using None")
            return None


# Additional utility functions for PyTorch

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: optimizer
        epoch: current epoch
        loss: current loss
        filepath: path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device='cuda'):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: optimizer
        filepath: path to checkpoint file
        device: device to load checkpoint on
    
    Returns:
        epoch, loss
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath}")
    return epoch, loss


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False