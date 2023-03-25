import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# declaring constants

A = 1.0
B = 663.0
C = 3.0
T_star = 12.0
u_star = -14.2
delta_x = 7.5

# making the vectorised differential equations

def f(state, t):
    u, T_e, T_w = state

    # setting up the 3 differential equations

    du = B / delta_x * (T_e - T_w) - C * (u - u_star) #  = du/dt
    dTe = u * T_w / (2 * delta_x) - A * (T_e - T_star) #  = dTe/dt
    dTw = -u * T_e / (2 * delta_x) - A * (T_w - T_star) #  = dTw/dt

    return du, dTe, dTw 


dt = .01
t_start = 0
t_end = 20

t = np.arange(t_start, t_end, dt)
state_0 = [10, 10, 14] # initial conditions
y = odeint(f, state_0, t) # y is a list with elemets that are list with 3 entries containing u, Te, Tw for each time step

# this corresponds to the functions with respect to time
u = y[:,0]
Te = y[:,1] 
Tw = y[:,2]

# first plots

plt.plot(t, u)
plt.title("Current velocity against time ({} years)".format(t_end))
plt.xlabel("Time t (years)")
plt.ylabel("Current Velocity u ") # 10^3km / year  units?
plt.ylim((-400,400))
plt.show()

plt.plot(t, Te - Tw)
plt.title("Difference in Temp against time ({} years)".format(t_end))
plt.xlabel("Time t (years)")
plt.ylabel("T_e - T_w ")
plt.ylim((-30,30))
plt.show()

# renaming variables to integrating for longer times

dt = .01
t_start = 0
t_end = 1000
t = np.arange(t_start, t_end, dt)
state_0 = [10, 10, 14] 
y = odeint(f, state_0, t)

u = y[:,0]
Te = y[:,1] 
Tw = y[:,2]

plt.plot(t, u)
plt.title("Current velocity against time ({} years)".format(t_end))
plt.xlabel("Time t (years)")
plt.ylabel("Current Velocity u ") # 10^3km / year  units?
plt.ylim((-400,400))
plt.show()
