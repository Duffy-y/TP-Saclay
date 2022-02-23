import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tp_lib import pente_extreme, incertitude_derivee_partielle
from sympy import *
from uncertainties import *

U, R, a1, a2, a3 = symbols("U, R, a1, a2, a3")
symbol1 = [U, R, a1]
symbol1_val = [12, 5, 0.015]
symbol1_incertitude = [0.1, 0, 0.001]

symbol2 = [U, R, a2]
symbol2_val = [6, 5, 0.004]
symbol2_incertitude = [0.1, 0, 0.001]

symbol3 = [U, R, a3]
symbol3_val = [12, 5, 0.009]
symbol3_incertitude = [0.1, 0, 0.001]

expr1 = U**2 / (R * a1)
expr2 = U**2 / (R * a2)
expr3 = U**2 / (R * a3)

CV1, dCV1 = incertitude_derivee_partielle(symbol1, symbol1_val, symbol1_incertitude, expr1)
CV2, dCV2 = incertitude_derivee_partielle(symbol2, symbol2_val, symbol2_incertitude, expr2)
CV3, dCV3 = incertitude_derivee_partielle(symbol3, symbol3_val, symbol3_incertitude, expr3)

c1 = ufloat(CV1, dCV1)
c2 = ufloat(CV2, dCV2)
c3 = ufloat(CV3, dCV3)

m1 = ufloat(0.45, 0.0001)
m3 = ufloat(0.8, 0.0001)

print(c1/m1)
print(c2/m1)
print(c3/m3)

print((c1/m1 + c2/m1 + c3/m3) / 3)



d_temp = 0.5
dt = 1

# Expérience 1.1 données
intensity_12V = 2.62 # A
current_12V = 12.0 # V

data_1 = np.genfromtxt('data/exp1.1.csv', delimiter=',', skip_header=1)
time_1 = data_1[:,1]
temperature_1 = data_1[:,2] - data_1[0,2]# K

# Expérience 1.2 données
intensity_6V = 1.32 # A
current_6V = 6.0 # V
data_2 = np.genfromtxt('data/exp1.2.csv', delimiter=',', skip_header=1)
time_2 = data_2[:,1]
temperature_2 = data_2[:,2] - data_2[0,2]# K

# Expérience 1.3 données
data_3 = np.genfromtxt('data/exp1.3.csv', delimiter=',', skip_header=1)
time_3 = data_3[:,1]
temperature_3 = data_3[:,2] - data_3[0,2]# K

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

ax1.set_ylabel('Différence de température (°C ou K)')
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
#plt.show()