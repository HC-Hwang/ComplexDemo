import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matrixDFT

import plotly
import plotly.graph_objs as go

import complex_demo

plotly.offline.init_notebook_mode()

shift = 10
sigma = 10
narray = 101

#generate a Gaussian function
x, gaussian_array = complex_demo.make1DGaussian(narray, sigma, x0 = shift)

#Fourier transform
#instantiate an mft object:
#make it as 2D array matrixDFT takes 2D
gaussian_array = np.array([gaussian_array])
ft = matrixDFT.MatrixFourierTransform()
ft_gaussian = ft.perform(gaussian_array, 9, (1, 81))

xx = np.linspace(-40, 40, 81)


#plot the result
complex_demo.complex_3dplot(xx, ft_gaussian[0], real_color='purple', plot=True, filename='example_gauss')
