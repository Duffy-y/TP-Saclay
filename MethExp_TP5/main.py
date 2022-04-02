from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.odr import *
from tp_lib import pente_extreme, incertitude_derivee_partielle
from sympy import *
from uncertainties import unumpy, ufloat

def BO(I, r, z):
    mu = 4 * np.pi * 10**(-7)
    N = 154
    return mu * N * I * r**2 / (2 * np.sqrt(r**2 + z**2)**3)

data = np.loadtxt("data/exp1.csv", delimiter=";", skiprows=1)
magn = data[:,0] * 10**(-3)
distance = data[:,1]
th = BO(1.999, 20 * 10**(-2), distance * 10**(-2))

data = np.loadtxt("data/exp1.2.csv", delimiter=";", skiprows=1)
magn = data[:,0] * 10**(-3)
courant = data[:,1]
th = BO(courant, 20 * 10**(-2), 0)

# plt.plot(distance, magn, 'o', label="Valeur expérimental")
# plt.plot(distance, th, 'r-', label="Loi de Biot-Savart")

plt.plot(courant, magn, 'o', label="Valeur exp")
plt.plot(courant, th, label="Loi de Biot-Savart")
plt.legend()
plt.show()