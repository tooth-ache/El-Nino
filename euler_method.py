#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:14:31 2023

@author: drozd
"""

import numpy as np
import matplotlib.pyplot as plt



"""
xmin = 0
xmax = 30
dx = .01
y_0 = -1/2

def f(x):
    return np.cos(x) / (x + 1)


x =  np.arange(xmin, xmax + dx, dx)
N = len(x)
y = np.zeros(N)
y[0] = y_0

for i in range(N - 1):
    y[i + 1] = y[i] + dx * f(x[i])

"""
"""plt.plot(x, y)
plt.show()
    """
# pendulum


def pendulum(y, t):
    omega = y[1]
    alpha = -np.sin(y[0])
    return np.array([omega, alpha])

tmin = 0
tmax = 100
dt = .01
t = np.arange(tmin,tmax,dt)
N = len(t)
y = np.zeros((N,2))
theta0 = np.pi/4
omega0 = 0
y[0,:] = [theta0, omega0]

for n in range(N-1):
    y[n+1,:] = y[n,:] + dt * pendulum(y[n,:],t[n])

plt.plot(t, y[:,0])
plt.show()