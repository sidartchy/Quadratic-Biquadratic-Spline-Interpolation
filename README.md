# Quadratic Spline Interpolation in Python

## Overview

This Python project implements quadratic spline interpolation, a mathematical technique used for creating a smooth curve that passes through a set of given data points. The project leverages the SymPy library for symbolic computation and provides a function for quadratic spline interpolation.

## Mathematical Background

### Quadratic Spline Interpolation

Quadratic spline interpolation involves dividing the dataset into segments and fitting a quadratic polynomial to each segment. The resulting piecewise quadratic function provides a smooth interpolation between data points.

The quadratic spline function for each segment $([x_i, x_i+1)$ can be represented as:

$(f_i(x) = a_i x^2 + b_i x + c_i)$

Where:
- \(a_i\), \(b_i\), and \(c_i\) are the coefficients specific to the i-th segment.
- \(x\) is the independent variable within the segment.


### System of Equations

To determine the coefficients \(a_i\), \(b_i\), and \(c_i\), a system of equations is derived. These equations ensure that the resulting piecewise function is not only continuous but also smooth at the junction points. This system of equations is solved using the SymPy library.

![quadratic spline interpolation](https://math.libretexts.org/@api/deki/files/109168/05.05.graph9.png?revision=1)


## Quadratic Spline Interpolation in Image Processing

Quadratic spline interpolation is a powerful technique in image processing for achieving smooth and visually pleasing results.


### Installation

```bash
pip install numpy matplotlib scipy sympy
```
## Acknowledgement

 - [Quadratic Spline Interpolation](https://shorturl.at/vALS8)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
