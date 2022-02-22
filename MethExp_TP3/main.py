import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tp_lib import pente_extreme, incertitude_derivee_partielle
from sympy import *


U, R, a = symbols("U, R, a")
lst_symbols = [U, R, a]
lst_symbols_value = [12, 5, 0.009]
lst_uncertainties = [0.1, 0, 0.001]

value, incertitude = incertitude_derivee_partielle(lst_symbols, lst_symbols_value, lst_uncertainties, U**2 / (R * a))
print(value, incertitude)

exit()

CV1, CV3, me1, me3 = symbols("CV1, CV3, me1, me3")
lst_symbols = [CV1, CV3, me1, me3]
lst_symbols_value = [1920, 3200, 0.45, 0.8]
lst_uncertainties = [160, 408, 0.0001, 0.0001]

expr_cvase = (CV1 * me3 - CV3 * me1)/(me3 - me1)

val_cvase, d_cvase = incertitude_derivee_partielle(lst_symbols, lst_symbols_value, lst_uncertainties, expr_cvase)

print(val_cvase, d_cvase)

CVase = Symbol("CVase")
lst_symbols = [CV3, CVase, me3]
lst_symbols_value = [3200, val_cvase, 0.8]
lst_uncertainties = [408, d_cvase, 0.0001]

expr_ceau = (CV3 - CVase) / me3

val_ceau, d_ceau = incertitude_derivee_partielle(lst_symbols, lst_symbols_value, lst_uncertainties, expr_ceau)
print(val_ceau, d_ceau)

exit()


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

A1, dA1, pente_max_1, pente_min_1 = pente_extreme(time_1, temperature_1, d_temp, dt)
A2, dA2, pente_max_2, pente_min_2 = pente_extreme(time_2, temperature_2, d_temp, dt)
A3, dA3, pente_max_3, pente_min_3 = pente_extreme(time_3, temperature_3, d_temp, dt)

print(A1, dA1)
print(A2, dA2)
print(A3, dA3)

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

temp1_regress = linregress(time_1, temperature_1)
temp2_regress = linregress(time_2, temperature_2)
temp3_regress = linregress(time_3, temperature_3)

temp1 = temp1_regress[0] * time_1 + temp1_regress[1]
temp2 = temp2_regress[0] * time_2 + temp2_regress[1]
temp3 = temp3_regress[0] * time_3 + temp3_regress[1]

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.set_ylabel('Température (K)')
ax1.set_xlabel("Temps [s]")
ax2.set_xlabel("Temps [s]")
ax3.set_xlabel("Temps [s]")

ax1.errorbar(time_1, temperature_1, xerr=1, yerr=0.5, fmt='go', label="12V 450gr")
ax1.plot(time_1, pente_max_1, 'g-')
ax1.plot(time_1, pente_min_1, 'g-')

ax2.errorbar(time_2, temperature_2, xerr=1, yerr=0.5, fmt='bo', label="6V 450gr")
ax2.plot(time_2, pente_max_2, 'b-')
ax2.plot(time_2, pente_min_2, 'b-')

ax3.errorbar(time_3, temperature_3, xerr=1, yerr=0.5, fmt='ro', label="12V 800gr")
ax3.plot(time_3, pente_max_3, 'r-')
ax3.plot(time_3, pente_min_3, 'r-')

ax1.legend()
ax2.legend()
ax3.legend()
# plt.show()