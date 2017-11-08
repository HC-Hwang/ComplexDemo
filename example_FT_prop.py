import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matrixDFT

import plotly
import plotly.graph_objs as go
from plotly import tools

from complex_demo import *

plotly.offline.init_notebook_mode()


sigma = 10
narray = 101

fig = tools.make_subplots(rows=8, cols=2,
                          specs=[[{'is_3d': True}, {'is_3d': True}] for i in range(8)])
ft = matrixDFT.MatrixFourierTransform()

x, gaussian_array = make1DGaussian(narray, sigma, x0 = 10)
gaussian_array = np.array([gaussian_array])
ft_gaussian = ft.perform(gaussian_array, 9, (1, 81))

x_data = np.linspace(-40, 40, 81)

func1 = lambda x, x0, sigma: np.exp(-(x-x0)**2./(2.*sigma**2.))
func2 = lambda x, x0, sigma: x * np.exp(-(x-x0)**2./(2.*sigma**2.))

x_input = np.linspace(-50, 50, 101)

y = func1(x_input, 0., sigma)
trace = complex_3dplot(x_input, y, plot=False, return_traces=True)
fig.append_trace(trace[0], 1, 1)
fig.append_trace(trace[1], 1, 1)

ft_y = ft.perform(np.array([y]), 9, (1, 81))
trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
fig.append_trace(trace[0], 1, 2)
fig.append_trace(trace[1], 1, 2)

y = func2(x_input, 0., sigma)
trace = complex_3dplot(x, y, plot=False, return_traces=True)
fig.append_trace(trace[0], 2, 1)
fig.append_trace(trace[1], 2, 1)

ft_y = ft.perform(np.array([y]), 9, (1, 81))
trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
fig.append_trace(trace[0], 2, 2)
fig.append_trace(trace[1], 2, 2)


y = func1(x_input, 0., sigma) * 1j
trace = complex_3dplot(x_input, y, plot=False, return_traces=True)
fig.append_trace(trace[0], 3, 1)
fig.append_trace(trace[1], 3, 1)

ft_y = ft.perform(np.array([y]), 9, (1, 81))
trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
fig.append_trace(trace[0], 3, 2)
fig.append_trace(trace[1], 3, 2)

y = func2(x_input, 0., sigma) * 1j
trace = complex_3dplot(x, y, plot=False, return_traces=True)
fig.append_trace(trace[0], 4, 1)
fig.append_trace(trace[1], 4, 1)

ft_y = ft.perform(np.array([y]), 9, (1, 81))
trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
fig.append_trace(trace[0], 4, 2)
fig.append_trace(trace[1], 4, 2)



y = func1(x_input, 0., sigma) * (1. + 1j)
trace = complex_3dplot(x_input, y, plot=False, return_traces=True)
fig.append_trace(trace[0], 5, 1)
fig.append_trace(trace[1], 5, 1)

ft_y = ft.perform(np.array([y]), 9, (1, 81))
trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
fig.append_trace(trace[0], 5, 2)
fig.append_trace(trace[1], 5, 2)

y = func2(x_input, 0., sigma) * (1. + 1j)
trace = complex_3dplot(x, y, plot=False, return_traces=True)
fig.append_trace(trace[0], 6, 1)
fig.append_trace(trace[1], 6, 1)

ft_y = ft.perform(np.array([y]), 9, (1, 81))
trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
fig.append_trace(trace[0], 6, 2)
fig.append_trace(trace[1], 6, 2)


y = func1(x_input, 10., sigma)
trace = complex_3dplot(x_input, y, plot=False, return_traces=True)
fig.append_trace(trace[0], 7, 1)
fig.append_trace(trace[1], 7, 1)

ft_y = ft.perform(np.array([y]), 9, (1, 81))
trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
fig.append_trace(trace[0], 7, 2)
fig.append_trace(trace[1], 7, 2)

y = func1(x_input, 10., sigma) * 1j
trace = complex_3dplot(x, y, plot=False, return_traces=True)
fig.append_trace(trace[0], 8, 1)
fig.append_trace(trace[1], 8, 1)

ft_y = ft.perform(np.array([y]), 9, (1, 81))
trace = complex_3dplot(x_data, ft_y[0], plot=False, return_traces=True)
fig.append_trace(trace[0], 8, 2)
fig.append_trace(trace[1], 8, 2)


fig['layout'].update(title='Properties of Fourier Transform',
                     height=1600, width=600)

plotly.offline.iplot(fig, filename='example_FT_prop')
