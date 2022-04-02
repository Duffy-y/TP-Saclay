from cProfile import label
from unittest import skip
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.odr import *
from tp_lib import pente_extreme, incertitude_derivee_partielle
from sympy import *

data1 = np.loadtxt("data/exp2.1.csv", delimiter=";", skiprows=1)
data2 = np.loadtxt("data/exp2.2.csv", delimiter=";", skiprows=1)
data3 = np.loadtxt("data/exp2.3.csv", delimiter=";", skiprows=1)


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))


ax[0].plot(data2[:,1], data2[:,0], 'o', label="Valeur expérimentale")

ax[1].plot(data1[:,1], data1[:,0], 'o', label="Valeur expérimentale")

ax[2].plot(data3[:,1], data3[:,0], 'o', label="Valeur expérimentale")

for axe in ax:
    axe.legend()
    axe.set_xlabel("Distance z [cm]")

ax[0].set_ylabel("B [mT]")
ax[0].set_title("Distance d = 10cm") 
ax[1].set_title("Distance d = 20cm") 
ax[2].set_title("Distance d = 50cm") 
    
fig.savefig("img/ChampDistance.eps", format="eps", dpi=1000)