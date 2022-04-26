import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.odr import *
from tp_lib import pente_extreme, incertitude_derivee_partielle
from sympy import *
import pandas as pd
from sklearn.metrics import r2_score

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

pression_chauffage = data_exp2_chauffage["Pression (hPa)"]
temperature_chauffage = data_exp2_chauffage["Température (K)"]
pression_refroid = data_exp2_refroidissement["Pression (hPa)"]
temperature_refroid = data_exp2_refroidissement["Température (K)"]


def linear(x, a, b):
    return a * x + b
    
popt, pcov = curve_fit(linear, temperature_chauffage, pression_chauffage, sigma=pression_chauffage * 0.01, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
print(perr)

pression_reg = popt[0] * temperature_chauffage + popt[1]

fig = plt.figure(figsize=(10, 5))
plt.errorbar(temperature_chauffage, pression_chauffage, yerr=pression_chauffage * 0.01, xerr=temperature_chauffage * 0.001 , fmt='o', label="Chauffage", zorder=-1, c="tab:red")
#plt.errorbar(temperature_refroid, pression_refroid, yerr=pression_refroid * 0.01, xerr=temperature_refroid * 0.001, fmt="bo", label="Refroidissement")
plt.plot(temperature_chauffage, pression_reg, "-", label="Régression linéaire", zorder=2, linewidth=3, c="tab:green")

print(f"R² score = {r2_score(pression_chauffage, pression_reg)}")
print(f"a = {popt[0]:.2f} ± {perr[0]:.2f}, b = {popt[1]:.2f} ± {perr[1]:.2f}")

print(f"{-popt[1]/popt[0]}")
print(f"{perr[1]/popt[0] + popt[1]/popt[0]**2 * perr[0]}")

plt.legend()
plt.xlabel("Température [K]")
plt.ylabel("Pression [hPa]")
#plt.savefig("img/exp2_graph2.eps", format="eps")
plt.show()
