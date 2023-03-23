import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

A = 1.0
B = 663.0
C = 3.0
T_star = 12.0
u_star = -14.2
delta_x = 7.5

def u(state, t):
    u, T_e, T_w = state
    return B / delta_x * (T_e - T_w) - C * (u - u_star)

def f(state, t):
    u, T_e, T_w = state
    return [  B / delta_x * (T_e - T_w) - C * (u - u_star),
            u * T_w / (2 * delta_x) - A * (T_e - T_star),
            -u * T_e / (2 * delta_x) - A * (T_w - T_star) ]


t = np.arange(0, 100, .01)
state_0 = [10, 10, 14]
y = odeint(f, state_0, t)

u = []
for i in range(len(y)):
    u.append(y[i][0])

dT = []
for i in range(len(y)):
    dT.append(y[i][1] - y[i][2])

plt.plot(t, u)
#plt.plot(t, dT)
#plt.plot(t, y[0,:], "--")
#plt.ylim((-400,400))
plt.show()
