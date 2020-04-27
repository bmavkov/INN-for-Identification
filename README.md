# Integrated Neural Networks for Nonlinear Continuous-Time System Identification

This repository contains the Python code to reproduce the results of the paper 
"Integrated Neural Networks for Nonlinear Continuous-Time System Identification" by Bojan Mavkov, Marco Forgione and Dario Piga.

The file codeThe main scripts are:

 *   `` main_CS``: rund the Cascade Tank example. (The cascade tank contains the dataset from the  [Cascaded Tanks System](http://www.nonlinearbenchmark.org/#Tanks))
 * ``NN_simulations.py``:  simulate the neural networks
 * ``NNmodels.py``: define  the neural networks 
 *   ``Norms.py``: contains the ERMS and R2 norms 


# Software requirements:
Simulations were performed on a Python 3.8 conda environment with

 * numpy
 * matplotlib
 * pandas
 * pytorch (version 1.5)
 
These dependencies may be installed through the commands:

```
conda install numpy numba scipy sympy pandas matplotlib ipython
conda install pytorch torchvision cpuonly -c pytorch
```

