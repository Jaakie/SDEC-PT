# sdecpt

PyTorch implementation of a version of Semi-supervised Deep Embedded Clustering (SDEC) algorithm, proposed by Ren et. al. in their paper "Semi-supervised deep embedded clustering ". The code is based on the PyTorch implementation of DEC by vlukiyanov (https://github.com/vlukiyanov/pt-dec). I only really made changes to the loss function, configuring it to use a constraint matrix.

## Examples

An example using MNIST data can be found in the `examples/mnist/mnist.py`.

## Usage

This is distributed as a Python package `sdecpt` and can be installed with `pip install .` after installing `ptsdae` from https://github.com/vlukiyanov/pt-sdae.

