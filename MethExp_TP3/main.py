import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.odr import *
from tp_lib import pente_extreme, incertitude_derivee_partielle
from sympy import *
from uncertainties import unumpy, ufloat


d_temp = 0.2
dt = 1

# Expérience 1.1 données
intensity_12V = 2.62 # A
current_12V = 12.0 # V

data_1 = np.genfromtxt('data/exp1.1.csv', delimiter=',', skip_header=1)
time_1 = data_1[:,1]
temperature_1 = data_1[:, 2] - data_1[0,2] # K

# Expérience 1.2 données
intensity_6V = 1.32 # A
current_6V = 6.0 # V
data_2 = np.genfromtxt('data/exp1.2.csv', delimiter=',', skip_header=1)
time_2 = data_2[:,1]
temperature_2 = data_2[:, 2] - data_2[0,2]  # K

# Expérience 1.3 données
data_3 = np.genfromtxt('data/exp1.3.csv', delimiter=',', skip_header=1)
time_3 = data_3[:,1]
temperature_3 = data_3[:, 2] - data_3[0,2]


A1, dA1, pente_max_1, pente_min_1 = pente_extreme(time_1, temperature_1, d_temp, dt)
A2, dA2, pente_max_2, pente_min_2 = pente_extreme(time_2, temperature_2, d_temp, dt)
A3, dA3, pente_max_3, pente_min_3 = pente_extreme(time_3, temperature_3, d_temp, dt)

print(A1, dA1)
print(A2, dA2)
print(A3, dA3)

# Expérience 2 données
boiling_water_temp = 94.6  # °C

laiton_mass_water = 451.1  # gr
laiton_mass_metal = 82.7  # gr
laiton_temp_initial = 21.1  # °C
laiton_temp_final = 22.4  # °C

teflon_mass_water = 451.1  # gr
teflon_mass_metal = 32.5  # gr
teflon_temp_initial = 22.3  # °C
teflon_temp_final = 23.4  # °C

plexi_mass_water = 450.0  # gr
plexi_mass_metal = 47.8  # gr
plexi_temp_initial = 22.3  # °C
plexi_temp_final = 23.4  # °C

duralumin_mass_water = 452.5  # gr
duralumin_mass_metal = 77.2  # gr
duralumin_temp_initial = 21.6  # °C
duralumin_temp_final = 24.0  # °C

# Expérience 3 données
data_4 = np.genfromtxt('data/exp2.csv', delimiter=',', skip_header=1)
time_4 = data_4[:,1]
temperature_4 = data_4[:, 2] # °C

log_temp = np.log(temperature_4)
error = np.mean(0.5 / temperature_4)

def model_func(B, x):
    return B[0] + B[1] * x

linear_model = Model(model_func)
data = RealData(time_4, log_temp, sy=error)
odr = ODR(data, linear_model, beta0=[0., 1.])
out = odr.run()

fit = out.beta[0] + out.beta[1] * time_4

print(out.beta)

plt.figure(figsize=[6, 4])
# plt.errorbar(time_4, temperature_4, yerr=0.2, fmt='o', label="Température de l'eau")
plt.errorbar(time_4, log_temp, yerr=error, fmt='o', label="Température de l'eau")
plt.plot(time_4, fit, 'r', label="Régression linéaire")
plt.xlabel("Temps (s)")
plt.ylabel("Logarithme de la température")
plt.legend()
#splt.savefig("img/Figure_3.png")