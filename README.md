# Integrated Neural Networks for Nonlinear Continuous-Time System Identification

This repository contains the Python code to reproduce the results of the paper 
"Integrated Neural Networks for Nonlinear Continuous-Time System Identification" by Bojan Mavkov, Marco Forgione and Dario Piga.

The main scripts are:

 *   ````: Symbolic manipulation of the RLC model, constant definition
 * ``RLC_genera.py``:  generate the identification dataset 
 * ``NNmodels.py``: generate 


# Software requirements:
Simulations were performed on a Python 3.8 conda environment with

 * numpy
 * scipy
 * matplotlib
 * pandas
 * sympy
 * numba
 * pytorch (version 1.5)
 
These dependencies may be installed through the commands:

```
conda install numpy numba scipy sympy pandas matplotlib ipython
conda install pytorch torchvision cpuonly -c pytorch
```

