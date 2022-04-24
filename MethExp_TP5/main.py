from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.odr import *
from tp_lib import pente_extreme, incertitude_derivee_partielle
from sympy import *

def BO(I, r, z):
    mu = 4 * np.pi * 10**(-7)
    N = 154
    return mu * N * I * r**2 / (2 * np.sqrt(r**2 + z**2)**3)



dB = 0.02
dz = 0.1 * 10**(-2)
dI = 0.005

R = 20 * 10**(-2)
dR = 0.1 * 10**(-2)

data1 = np.loadtxt("data/exp1.csv", delimiter=";", skiprows=1)
B1 = data1[:,0]
z1 = data1[:,1] * 10 ** (-2)
th1 = BO(2.000, R, z1) * 10**(3)

data2 = np.loadtxt("data/exp1.2.csv", delimiter=";", skiprows=1)
B2 = data2[:,0]
I2 = data2[:,1]
th2 = BO(I2, R, 0) * 10**3

fig = plt.figure(figsize=(7, 4))

size = len(z1)
B_moy = list(np.ones(len(z1)))
B = list(B1)
for i in range(18):
    val = (B[i] + B[size - 1 - i]) / 2
    B_moy[i] = val
    B_moy[size - 1 - i] = val
    

plt.errorbar(z1, th1, yerr=0.01, xerr=dz, fmt='r-', label="Loi de Biot-Savart")
plt.errorbar(z1, B1, yerr=dB, xerr=dz, fmt='o', label="Valeur expérimentales")
plt.errorbar(z1, B_moy, yerr=dB, xerr=dz, fmt='b-', label="Valeur moyenne")

#plt.errorbar(I2, B2, yerr=dB, xerr=dI, fmt='o', label="Valeur expérimentales")
#plt.errorbar(I2, th2, yerr=0.006, xerr=dI, label="Loi de Biot-Savart")

#ecart = np.abs(B2 - th2)
#plt.plot(I2, ecart/np.max(ecart), 'r-', label="Ecart relatif")

plt.xlabel("Distance z [m]")
plt.ylabel("Intensité du champ magnétique [mT]")
plt.legend()
#plt.show()
fig.savefig('img/Moyenne.eps', format="eps", dpi=1000)