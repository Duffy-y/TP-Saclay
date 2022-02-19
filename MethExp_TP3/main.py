import numpy as np
import matplotlib.pyplot as plt
from uncertainties import *

d_temp = 0.5
dt = 1

# Expérience 1.1 données
intensity_12V = 2.62 # A
current_12V = 12.0 # V

data_1 = np.genfromtxt('data/exp1.1.csv', delimiter=',', skip_header=1)
time_1 = data_1[:,1]
temperature_1 = data_1[:,2] + 273.15 # K

# Expérience 1.2 données
intensity_6V = 1.32 # A
current_6V = 6.0 # V
data_2 = np.genfromtxt('data/exp1.2.csv', delimiter=',', skip_header=1)
time_2 = data_2[:,1]
temperature_2 = data_2[:,2] + 273.15 # K

# Expérience 1.3 données
data_3 = np.genfromtxt('data/exp1.3.csv', delimiter=',', skip_header=1)
time_3 = data_3[:,1]
temperature_3 = data_3[:,2] + 273.15 # K

A_max = (temperature_2[-1] + d_temp - (temperature_2[0] - d_temp)) / (time_2[-1] - time_2[0])
A_min = (temperature_2[-1] - d_temp - (temperature_2[0] + d_temp)) / (time_2[-1] - time_2[0])

A = (A_max + A_min) / 2
dA = (A_max - A_min) / 2

uA = ufloat(A, dA)

print(uA)

cV = (intensity_6V * current_6V) / (uA * 450.0 * 10**(-3))

print(cV)


# Expérience 2 données
boiling_water_temp = 94.6 + 273.15 # K

laiton_mass_water = 451.1 * 10**(-3) # gr
laiton_mass_metal = 82.7 * 10**(-3) # gr
laiton_temp_initial = 21.1 + 273.15 # K
laiton_temp_final = 22.4 + 273.15 # K

teflon_mass_water = 451.1 * 10**(-3) # gr
teflon_mass_metal = 32.5 * 10**(-3) # gr
teflon_temp_initial = 22.3 + 273.15 # K
teflon_temp_final = 23.4 + 273.15 # K

plexi_mass_water = 450.0 * 10**(-3) # gr
plexi_mass_metal = 47.8 * 10**(-3) # gr
plexi_temp_initial = 22.3 + 273.15 # K
plexi_temp_final = 23.4 + 273.15 # K

duralumin_mass_water = 452.5 * 10**(-3) # gr
duralumin_mass_metal = 77.2 * 10**(-3) # gr
duralumin_temp_initial = 21.6 + 273.15 # K
duralumin_temp_final = 24.0 + 273.15 # K

plt.errorbar(time_1, temperature_1, xerr=1, yerr=0.5, fmt='go', label="12V 450gr")
plt.errorbar(time_2, temperature_2, xerr=1, yerr=0.5, fmt='bo', label="6V 450gr")
plt.errorbar(time_3, temperature_3, xerr=1, yerr=0.5, fmt='ro', label="12V 800gr")
plt.xlabel("Temps [s]")
plt.ylabel("Température [K]")
plt.legend()
plt.show()