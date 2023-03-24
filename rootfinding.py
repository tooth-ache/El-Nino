#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:11:49 2023

@author: drozd
"""

import numpy as np

def func1(x):  # root at x = 1.709975242614746
    return x**3 - 5


def func2(x): # root at x = 0.7953996658325195
    return np.cos(x) - .7

def bisect(a, b, func):
    x1 = 0
    x2 = 0
    x3 = 0
    if func(a) < 0:
        x1 = a
        x2 = b
    else:
        x1 = b
        x2 = a

    run = True
    times_run = 0
    while run:
        times_run += 1
        x3 = (x1 + x2)/2

        if func(x3) < 0:
            x1 = x3
        else:
            x2 = x3
        if np.abs(x1 - x2) < 10**(-6):
            run = False

    return x3, times_run

def false_pos(a, b, func):
    x1 = a
    x2 = b

    times_run = 0


    run = True
    while run:
        y1 = func(x1)
        x1_prev = x1
        y2 = func(x2)
        times_run += 1
        x1 = x2 - y2 * (x2 - x1) / (y2 - y1)
        if np.abs(x1 - x1_prev) < 10**(-6):
            run = False
    return x1, times_run



my_num = false_pos(0, 2, func1)
print(my_num)