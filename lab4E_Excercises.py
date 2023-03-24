import numpy as np
import math

def poisson(n, l):
    if type(n) is not int:
        print("n is an interger bigger than 0")
        return
    if n < 0:
        print("n is an interger bigger than 0")
        return
    else:
        f = (l**n)*(np.exp(-l))/(np.math.factorial(n))
        return f


def g(x, N):
    result = 0
    for n in range(1, N):
        result += (np.sin(x)**n) / (2 * (n**2) )
    return result

# now integration with trapezoid rule

def function(x):
    return (x**4 - 3 * x**2)

delta = .001
min_lim = -1
max_lim = 2

x = np.arange(min_lim, max_lim, delta)
lenght = len(x)
f = function(x)
result = 0
for i in range(0, lenght -1):
    result += (f[i] + f[i + 1]) * (delta/2)

print(result)