import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matrixDFT

import plotly
import plotly.graph_objs as go

import complex_demo

plotly.offline.init_notebook_mode()

#shifts will be the changing variable in the animation
shifts = range(20)

sigma = 10
narray = 101

#x axis
x_data = np.linspace(-40, 40, 81)

#for each shift, store the data in the list: complex_data
complex_data = []
for shift in shifts:
    x, gaussian_array = complex_demo.make1DGaussian(narray, sigma, x0 = shift)
    gaussian_array = np.array([gaussian_array])
    ft = matrixDFT.MatrixFourierTransform()
    ft_gaussian = ft.perform(gaussian_array, 9, (1, 81))
    complex_data.append(ft_gaussian[0]) #be careful of [0]

complex_demo.complex_3danimation(x_data, complex_data, shifts, line_color='purple', plot=True, filename='example_gauss_ani')


