from uncertainties import ufloat, unumpy
import numpy as np
import matplotlib.pyplot as plt
import ldd


V = unumpy.uarray([0, 2.43, 6.65, 8.054, 9.115, 10.187, 11.656, 12.590, 13.300, 13.846, 14.300, 14.366, 15.007], 0.2)
r = unumpy.uarray([0.75, 0.83, 1.76, 2.38, 2.85, 3.43, 4.32, 4.96, 6.09, 6.76, 6.94, 7.84, 7.90], 0.1)

R_1 = ufloat(0.75, 0.05)
R_2 = ufloat(8, 0.05)

log_r = unumpy.log(r/R_1)

(a, b) = ldd.odr_fit(ldd.linear_function, log_r, V, (1, 0))

ldd.plot(log_r, ldd.linear_function([a, b], log_r), label="ODR")
ldd.plot_error(log_r, V, label="exp", fmt="r.")
plt.legend()
plt.show()