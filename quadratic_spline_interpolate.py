import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sympy import *

def quadratic_interpolation(x_data,y_data,x_new,plot=False):
  """
    Perform quadratic interpolation on a set of data points and optionally plot the interpolated curve.

    Parameters:
    - x_data (array-like): x-coordinates of the input data points.
    - y_data (array-like): y-coordinates of the input data points.
    - x_new (float): x-coordinate for which the interpolation is desired.
    - plot (bool, optional): If True, the function will plot the interpolated curve along with the input data points.

    Returns:
    - solution (dict): Coefficients of the quadratic interpolating polynomials.
    - yn (float): Interpolated y-value corresponding to x_new.

    The function uses symbolic computation to derive a system of equations based on the quadratic polynomial form.
    It then solves the system of equations to obtain the coefficients for each quadratic segment.

    If 'plot' is set to True, the function will display a plot of the interpolated curve along with the input data points.
    The interpolated y-value at x_new is also marked on the plot.

    Example:
    ```python
    x_data = [1, 2, 3, 4]
    y_data = [2, 3, 5, 10]
    x_new = 2.5
    solution, yn = quadratic_interpolation(x_data, y_data, x_new, plot=True)
    ```

    Note: This function assumes that the input data points are provided in ascending order of x-coordinates.

  """
  data = [[i,j] for i,j in zip(x_data,y_data)]
  points = np.array(data)
  n = len(points) - 1

  x, y = symbols('x, y')
  a = symbols('a1:%d'%(n+1))
  b = symbols('b1:%d'%(n+1))
  c = symbols('c1:%d'%(n+1))

  f = [a[i]*x**2 + b[i]*x + c[i] - y for i in range(n)]


  equations = []
  equations.append(f[0].subs(x, points[0, 0]).subs(y, points[0, 1]))


  for i in range(n - 1):
    equations.append(f[i].subs(x, points[i + 1, 0]).subs(y, points[i + 1, 1]))
    equations.append(f[i + 1].subs(x, points[i + 1, 0]).subs(y, points[i + 1, 1]))

  equations.append(f[-1].subs(x, points[-1, 0]).subs(y, points[-1, 1]))

  fdx = [diff(fi, x) for fi in f]
  for i in range(n - 1):
    equations.append(fdx[i].subs(x, points[i + 1, 0]) - fdx[i + 1].subs(x, points[i + 1, 0]))


  equations.append(a[-1])
  print(equations)

  equation_tuple = tuple(equations)
  print(equation_tuple)

  coef_tuple = tuple(a+b+c)
  print(coef_tuple)

  solution = solve(equation_tuple, coef_tuple)
  print(solution)

  if (plot == True):
    for i in range(n):
        span = np.linspace(points[i, 0], points[i + 1, 0], 100)
        fi = f[i].subs(solution)
        print(fi)
        plt.plot(span, [solve(fi.subs(x, i)) for i in span], label='f{i}'.format(i=i))
    plt.scatter(points[:, 0], points[:, 1])


  for i in range(n):
    if x_new >= points[i,0] and x_new <= points[i+1,0]:
      fi=  f[i].subs(solution)
      yn = solve(fi.subs(x,x_new))
      plt.scatter(x_new, yn,label='f{i}(x_new)'.format(i=i))
      break;
    else:
      yn = 0

  if(yn ==0):
    print("The value is out of bound")

  plt.grid()
  plt.legend()
  plt.show()
  return solution,yn