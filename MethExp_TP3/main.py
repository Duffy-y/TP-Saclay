import numpy as np
import matplotlib.pyplot as plt
from uncertainties import *
from scipy.stats import linregress
from sympy import *

from typing import Any, List, Tuple

d_temp = 0.5
dt = 1

def pente_extreme(x, y, dy) -> Any:
    A_max = (y[-1] + dy - (y[0] - dy)) / (x[-1] - x[0])
    A_min = (y[-1] - dy - (y[0] + dy)) / (x[-1] - x[0])

    A = (A_max + A_min) / 2
    dA = (A_max - A_min) / 2
    
    return A, dA, A_max * x + y[0] - dy, A_min * x + y[0] + dy

dU = 0.1

U1 = 12
R1 = 5
a1 = 0.015
da1 = 0.001
C_V1 = U1**2 / (R1 * a1)
dCV1 = U1**2 / (R1 * a1 **2) * da1 + (2 * U1)/(R1 * a1) * dU

U3 = 12
R3 = 5
a3 = 0.087
da3 = 0.001
C_V3 = U3**2 / (R3 * a3)
dCV3 = U3**2 / (R3 * a3 **2) * da3 + (2 * U3)/(R3 * a3) * dU


m_eau3 = 0.8
m_eau1 = 0.45
dme3 = 0.0001
dme1 = 0.0001

cvase = C_V3

vase_partial_cv1 = abs(m_eau3/(-m_eau1 + m_eau3)) * dCV1
vase_partial_cv3 = abs(-m_eau1/(-m_eau1 + m_eau3)) * dCV3
vase_partial_me3 = abs(C_V1/(-m_eau1 + m_eau3) - (C_V1*m_eau3 - C_V3*m_eau1)/(-m_eau1 + m_eau3)**2) * dme3
vase_partial_me1 = abs(-C_V3/(-m_eau1 + m_eau3) + (C_V1*m_eau3 - C_V3*m_eau1)/(-m_eau1 + m_eau3)**2) * dme1

delta_vase = vase_partial_cv1 + vase_partial_cv3 + vase_partial_me3 + vase_partial_me1

eau_partial_cv3 = abs(1/m_eau3) * dCV3
eau_partial_cvase = abs(-1/m_eau3) * delta_vase
eau_partial_me3 = abs(-(C_V3 - cvase)/m_eau3**2) * dme3

delta_ceau = eau_partial_cv3 + eau_partial_cvase + eau_partial_me3

print()

print(delta_vase)
print(delta_ceau)

exit()

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

A1, dA1, pente_max_1, pente_min_1 = pente_extreme(time_1, temperature_1, d_temp)
A2, dA2, pente_max_2, pente_min_2= pente_extreme(time_2, temperature_2, d_temp)
A3, dA3, pente_max_3, pente_min_3 = pente_extreme(time_3, temperature_3, d_temp)

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