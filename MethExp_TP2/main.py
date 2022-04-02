import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.odr import *
from tp_lib import pente_extreme, incertitude_derivee_partielle
from sympy import *

def chiffre_cs(x, u):
    if u == 0:
        return ""
    x = abs(x)
    u = abs(u)
    apres_virgule = 0
    val = u
    while str(val)[0] == "0":
        apres_virgule += 1
        val *= 10
    return f"{round(x, apres_virgule)} $\\pm$ {round(u, apres_virgule)}"

# uncertainty
dt = 0.033 #s
dm = 0.001 # kg

m1 = 0.5286 #kg
m2 = 0.5282 #kg 

mass_dict = {0: 0, 1: 0.01, 2: 0.05, 3: 0.497, 4: 0.5021} # kg

rapport_mass = []
rapport_mass_linspace = np.linspace(0.5, 2, 50)

initial_velocity_1 = []
initial_velocity_2 = []
final_velocity_1 = []
final_velocity_2 = []

d_ini_v1 = []
d_fin_v1 = []
d_fin_v2 = []

initial_kinetic_1 = []
initial_kinetic_2 = []
final_kinetic_1 = []
final_kinetic_2 = []

d_ini_ec1 = []
d_fin_ec1 = []
d_fin_ec2 = []

R_exp = []
R_err = []

R_th = 4 * rapport_mass_linspace * (1 + rapport_mass_linspace) ** (-2) * 100

# CHARGE TOUTES LES PUTAINS DE COLLISIONS AHHHHHH
for i in range(1, 13):
    print(f"Loading data {i}")
    data = np.loadtxt("data/Feuille" + str(i) + ".csv", delimiter=";", skiprows=1)
    
    # Data collection
    t = data[:, 0]
    x1 = data[:, 1]
    x2 = data[:, 2]
    total_m1 = m1 + mass_dict[data[0, 3]]
    total_m2 = m2 + mass_dict[data[0, 4]]
    
    # Speed calculation
    v1 = (x1[1:] - x1[:-1]) / dt
    v2 = (x2[1:] - x2[:-1]) / dt
    
    _, dvi_1, _, _ = pente_extreme(t[:4], x1[:4], 0.005, dt)
    _, dvf_1, _, _ = pente_extreme(t[-4:], x1[-4:], 0.005, dt)
    _, dvf_2, _, _ = pente_extreme(t[-4:], x2[-4:], 0.005, dt)
    
    initial_v1 = np.mean(v1[:4])
    initial_v2 = 0
    final_v1 = np.mean(v1[-4:])
    final_v2 = np.mean(v2[-4:])
    
    rapport_mass.append(total_m1/total_m2)
    initial_velocity_1.append(initial_v1)
    initial_velocity_2.append(initial_v2)
    final_velocity_1.append(final_v1)
    final_velocity_2.append(final_v2)
    
    d_ini_v1.append(dvi_1)
    d_fin_v1.append(dvf_1)
    d_fin_v2.append(dvf_2)
    
    initial_ec_1 = 1/2 * total_m1 * initial_v1 ** 2
    initial_ec_2 = 1/2 * total_m2 * initial_v2 ** 2
    final_ec_1 = 1/2 * total_m1 * final_v1 ** 2
    final_ec_2 = 1/2 * total_m2 * final_v2 ** 2
    
    initial_kinetic_1.append(initial_ec_1)
    initial_kinetic_2.append(initial_ec_2)
    final_kinetic_1.append(final_ec_1)
    final_kinetic_2.append(final_ec_2)
    
    deci_1 = initial_v1 ** 2 / 2 * dm + total_m1 * initial_v1 * dvi_1
    decf_1 = final_v1 ** 2 / 2 * dm + total_m1 * final_v1 * dvf_1
    decf_2 = final_v2 ** 2 / 2 * dm + total_m2 * final_v2 * dvf_2
    
    d_ini_ec1.append(deci_1)
    d_fin_ec1.append(decf_1)
    d_fin_ec2.append(decf_2)
    
    R_exp.append(final_ec_2 / initial_ec_1 * 100)
    R_err.append(decf_2 / deci_1 )
    

latex = ""
for i, (r, vi1, vf1, vf2, dvi1, dvf1, dvf2, eci1, ecf1, ecf2, deci1, decf1, decf2) in enumerate(zip(rapport_mass, initial_velocity_1, final_velocity_1, final_velocity_2, d_ini_v1, d_fin_v1, d_fin_v2, initial_kinetic_1, final_kinetic_1, final_kinetic_2, d_ini_ec1, d_fin_ec1, d_fin_ec2)):
    latex += f"{r:.3f} & {chiffre_cs(vi1, dvi1)} & {chiffre_cs(vf1, dvf1)} & {chiffre_cs(vf2, dvf2)} & "
    print(i)
    latex += f"{chiffre_cs(eci1, deci1)} & {chiffre_cs(ecf1, decf1)} & {chiffre_cs(ecf2, decf2)} \\\\ \n"
    
# output latex in a file
with open("data/table12.txt", "w") as f:
    f.write(latex)
    
plt.errorbar(rapport_mass, R_exp, fmt="o", yerr=9, label="Résultat expérimental")
plt.plot(rapport_mass_linspace, R_th, "-", label="Résultat théorique")
plt.xlabel("Rapport de masse")
plt.ylabel("Coefficient de restitution (%)")
plt.legend()
plt.show()
