import matplotlib.pyplot as plt
import numpy as np

delta = .1

x = np.arange(0, 10, delta)
c = np.cos(x)
f = np.sin(x)

dff = np.zeros(len(x) - 1)
#print(len(x) - 1)

for i in range(len(x) - 1):
    dff[i] = (f[i + 1] - f[i]) / delta

dfb = np.zeros(len(x) - 1)

for i in range(1, len(x)):
    dfb[i-1] = (f[i] - f[i - 1]) / delta

dfs = np.zeros(len(x) - 2)

for i in range(1, len(x) - 1):
    dfs[i-1] = (f[i + 1] - f[i - 1]) / (2 * delta)

plt.plot(x[:-1], dff, "--", label = "dff")
plt.plot(x[1:], dfb, "--", label = "dfb")
plt.plot(x[1:-1], dfs, "--", label = "dfs")
plt.legend()

plt.plot(x, y)
#plt.plot(x, c)
plt.show()