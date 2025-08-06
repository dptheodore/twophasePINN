# Physics-informed neural networks for two-phase flow problems
Physics-informed neural networks (PINN) give rise to a new approach for the quantification of flow fields
by combining available data of auxiliary variables with the underlying physical laws. This way, the
velocity and pressure of entire flow fields may be inferred, for which a direct measurement is usually
impracticle.

This repository contains the code for the rising bubble case in the paper "Inferring incompressible two-phase flow fields from the interface
motion using physics-informed neural networks" by Aaron B. Buhendwa, Stefan Adami and Nikolaus A. Adams.

If there are any questions regarding the code please contact us by [mail](mailto:aaron.buhendwa@tum.de).
# Prerequisites
Before running the scripts, the file containing the [CFD result](https://syncandshare.lrz.de/getlink/fiQgX8w2H3UhNrYqHxEPV8/rising_bubble.h5) has to be downloaded and put into the folder `cfd_data`. Furthermore, your python environment must have the following packages installed:
* numpy, scipy, pandas, mat4py, tensorflow, h5py, matplotlib 

# Running the scripts
We provide two scripts that are running out of the box located in `src`:

* `rising_bubble_train.py` contains the implementation of the PINN class and the training routine. When running this script, the point distribution is generated and displayed for multiple time snapshots. Subsequently, the PINN is instantiated and trained using the (default) hyperparameters as described in the paper. Note that when using the default hyperparameters and amount of training points, a single epoch takes about 4 seconds on a GeForce RTX 2080Ti and thus may take substantially longer when running on a CPU. In this case the network size and/or amount of training points should be reduced by the user by setting the corresponding variables within the `main` function. During training, this script will generate a new folder called `checkpoints`, where, at user defined epoch intervals, the model is saved. Please refer to the respective function descriptions for further details.

* `rising_bubble_test.py` contains a test environment that loads both the CFD result and a PINN. The prediction is then compared to the cfd by 
animating contourplots of the velocity and pressure. By default, this script will load the PINN that is located in the directory `checkpoints`.

* `poiseuille_train.py` contains a PINN implementation of the forward poiseuille flow problem. At the top of the file, you can manually set mu1, which will set up your viscosity ratios, as mu2 is already at 1.0. Furthermore, you can change adaptive activation to True to use an adaptive activation function initialized with a=0.1 per neuron and n=10 as per the papers instructions.

* `poiseuille_test.py` simply shows a comparison between the analytical solution and the predicted solution after running the training script.

* For the poiseuille flow scripts, everything pertaining to those files is saved to the `pinn_output` directory based on your choice of `mu1` and `adaptive_activation`. This means that best weights will be overwritten per mu and adaptive activation choice.

# Citation
DOI: 10.1016/j.mlwa.2021.100029
