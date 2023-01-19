import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat, unumpy
from scipy.odr import RealData, Model, ODR
from typing import Callable

def linear_function(beta, x: float | np.ndarray):
    """Returns f(x) = a * x + b
    
    Args:
        beta (Tuple[float, float]): Iterable containing slope and y-intercept in this exact order
        x (float or np.ndarray): x-coordinate or list of x-coordinates
    """
    return beta[0] * x + beta[1]

def exponential_decay(beta, x: float | np.ndarray) -> float | np.ndarray:
    """Returns amplitude * exp(-x / tau) + B

    Args:
        beta (Tuple[float, float, float]): Iterable containing amplitude, tau and B in this exact order 
        x (float|np.ndarray): x-coordinates

    Returns:
        float or np: y-values at x-coordinates of the given function
    """

    return beta[0] * np.exp(- x / beta[1]) + beta[2]

def odr_fit(model: Callable, x: np.ndarray, y: np.ndarray, p0, fit_type = 0):
    """Performs an OLS/ODR onto a dataset of given x-y coordinates.

    Args:
        model (Callable): Function to fit
        x (np.ndarray): x-coordinates
        y (np.ndarray): y-coordinates
        p0 (_type_): _description_
        fit_type (int, optional): 2 = least square, 0 = orthogonal direction regression. Defaults to 0.

    Returns:
        list[ufloat]: list of fitted parameters returned as ufloat
        
    Examples:
        V = unumpy.uarray([0, 2.43, 6.65, 8.054, 9.115, 10.187, 11.656, 12.590, 13.300, 13.846, 14.300, 14.366, 15.007], 0.2)
        r = unumpy.uarray([0.75, 0.83, 1.76, 2.38, 2.85, 3.43, 4.32, 4.96, 6.09, 6.76, 6.94, 7.84, 7.90], 0.1)
        R_1 = ufloat(0.75, 0.05)
        R_2 = ufloat(8, 0.05)
        log_r = unumpy.log(r/R_1)
        
        (a, b) = odr_fit(linear_function, log_r, V, (1, 0)) # odr_fit will auto extract uncertainties and return value with uncertainty
    """
    # Extracts values and uncertainties from uncertainties package or fallback on list based value and uncertainity
    x_val = unumpy.nominal_values(x)
    x_err = unumpy.std_devs(x)
    y_val = unumpy.nominal_values(y)
    y_err = unumpy.std_devs(y)
    
    # Orthogonal direction regression 
    data = RealData(x_val, y_val, sx=x_err, sy=y_err)
    model = Model(fcn=model)
    odr = ODR(data, model, beta0=p0)
    odr.set_job(fit_type=fit_type)
    output = odr.run()
    
    out = []
    for parameter, err in zip(output.beta, output.sd_beta):
        out.append(ufloat(parameter, err))
        
    return out

def plot_error(x, y, label="", fmt=None):
    if fmt == None:
        plt.errorbar(x=unumpy.nominal_values(x), y=unumpy.nominal_values(y), xerr=unumpy.std_devs(x), yerr=unumpy.std_devs(y), label=label)
    else:
        plt.errorbar(x=unumpy.nominal_values(x), y=unumpy.nominal_values(y), xerr=unumpy.std_devs(x), yerr=unumpy.std_devs(y), label=label, fmt=fmt)

V = unumpy.uarray([0, 2.43, 6.65, 8.054, 9.115, 10.187, 11.656, 12.590, 13.300, 13.846, 14.300, 14.366, 15.007], 0.2)

r = unumpy.uarray([0.75, 0.83, 1.76, 2.38, 2.85, 3.43, 4.32, 4.96, 6.09, 6.76, 6.94, 7.84, 7.90], 0.1)

R_1 = ufloat(0.75, 0.05)
R_2 = ufloat(8, 0.05)

log_r = unumpy.log(r/R_1)

(a, b) = odr_fit(linear_function, log_r, V, (1, 0))

plot_error(log_r, linear_function([a, b], log_r), label="ODR", fmt="r-")
plt.plot(log_r, V, label="exp", c="blue")
plt.legend()
plt.show()
    