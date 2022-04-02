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

mu, N, I, r, z = symbols('mu, N, I, r, z')
dMu, dN, dI, dR, dZ = [0, 0, dI, dR, dz]

expression = mu * N * I * r**2 / (2 * (r**2 + z**2)**(3/2))

B = []
dBF = []
for i in I2:
    values = [4 * np.pi * 10**(-7), 154, i, R, 0]
    val, err = incertitude_derivee_partielle([mu, N, I, r, z], values, [dMu, dN, dI, dR, dZ], expression)
    B.append(val)
    dBF.append(err)

dBF = np.array(dBF) * 10**(3)
print(dBF)

#plt.plot(z1, th1, 'r-', label="Loi de Biot-Savart")
plt.errorbar(z1, B1, yerr=dB, xerr=dz, fmt='o', label="Valeur expérimentales")

#plt.errorbar(I2, B2, yerr=dB, xerr=dI, fmt='o', label="Valeur expérimentales")
#plt.errorbar(I2, th2, yerr=dBF, label="Loi de Biot-Savart")

#ecart = np.abs(B2 - th2)
#plt.plot(I2, ecart/np.max(ecart), 'r-', label="Ecart relatif")

plt.xlabel("Distance z [m]")
plt.ylabel("Intensité du champ magnétique [mT]")
plt.legend()
#plt.show()
fig.savefig('img/MagnEnFonctionDeZ.eps', format="eps", dpi=1000)