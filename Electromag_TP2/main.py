import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./Electromag_TP2/angle.csv", sep=";")

theta = df["Angle"]
dtheta = 5
U2 = df["U2"]
dU2 = df["U2"] * 0.03
cos_theta = np.cos(theta * np.pi / 180)
dcos = np.sin(theta * np.pi / 180) * dtheta

plt.plot(cos_theta, U2, "o")
plt.xlabel("cos(theta)")
plt.ylabel("Tension (V)")
plt.show()