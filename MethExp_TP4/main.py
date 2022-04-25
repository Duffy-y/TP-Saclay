import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.odr import *
from tp_lib import pente_extreme, incertitude_derivee_partielle
from sympy import *
import pandas as pd

def linear(x, a, b):
    return a * x + b

# Experience 1
dV = 1 * 10**(-6) #m³
dP = 1 * 10**(2) # Pa
# Incertitude pression = 2% de la valeur
data_exp1 = pd.read_csv("data/Exp1.csv", delimiter=";")

V_tuyau = 0.942 * 10 ** (-6) # m³

# Expérience 2
V_bocal = 250 * 10**(-6) # m³
dT = 0.1
# Incertitude température = 0.2% de la valeur
dP = 1 * 10**(2) # Pa
data_exp2_chauffage = pd.read_csv("data/Exp2Chauffage.csv", delimiter=";")
data_exp2_refroidissement = pd.read_csv("data/Exp2Refroidissement.csv", delimiter=";")

plt.plot(data_exp2_chauffage["Température (K)"], data_exp2_chauffage["Pression (hPa)"], 'ro', label="Chauffage")
plt.plot(data_exp2_refroidissement["Température (K)"], data_exp2_refroidissement["Pression (hPa)"], "bo", label="Refroidissement")

plt.legend()
plt.xlabel("Température [K]")
plt.ylabel("Pression [hPa]")
plt.show()
