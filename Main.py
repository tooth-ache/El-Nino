import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# in the first half, we show all of the functions we will use in the program because it think it will be easier to read.
# Then we will have all of the plots and results we want to obtain in the second half.

""" FIRST HALF: DECLARING FUNCTIONS """

# making a function for the set of differential equations
def system(state, t):
    u, T_e, T_w = state

    # constants, to the right of each constant i noted its original value
    A = 1.0 # 1.0
    B = 663.0 # 663.0
    C = 3.0 # 3.0 
    T_star = 12.0 # 12.0
    u_star = -14.2 # -14.2
    delta_x = 7.5 # 7.5

    # setting up the 3 differential equations
    du = B / delta_x * (T_e - T_w) - C * (u - u_star) #  = du/dt
    dTe = u * T_w / (2 * delta_x) - A * (T_e - T_star) #  = dTe/dt
    dTw = -u * T_e / (2 * delta_x) - A * (T_w - T_star) #  = dTw/dt

    return du, dTe, dTw

# function for taking the derivative of a function (symmetric derivative)
def diff(func, x, dx):
    dfs = np.zeros(len(x) - 2)
    for i in range(1, len(x) - 1):
        dfs[i-1] = (func[i + 1] - func[i - 1]) / (2 * dx)
    return dfs

# this will turn the numpy array into a something that works like a mathematical function
def make_func(list_func, x):
    if len(list_func) - 1 < int(round(x/dt)): # this bit makes sure that the 
        return list_func[-1]
    else:
        return list_func[int(round(x/dt))]

# function for root finding (bisection method)
def find_root(a, b, func):
    x1 = 0
    x2 = 0
    x3 = 0
    if make_func(func, a) * make_func(func, b) > 0:
        x3 = "None" # im makeing the output a string if there is no root
        run = False

    if make_func(func, a) < 0:
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
        if make_func(func, x3) < 0:
            x1 = x3
        else:
            x2 = x3
        if np.abs(x1 - x2) < 10**(-3): # precison
            run = False
        elif times_run > 100:
            run = False
            x3 = "None"  # im makeing the output a string if there is no root

    return x3

# sexy root finder
def find_all_maxima(f, df, start_step, end_step, step_size, y_shift):
    roots = []
    brackets = []
    a = start_step
    b = a + step_size
    brackets_start = 0
    for i in range(int(end_step/step_size)):
                
        if (make_func(f, a) - y_shift) * (make_func(f, b) - y_shift) < 0:
            if len(brackets) == 0:
                if (make_func(f, a) - y_shift) > 0:
                    brackets_start = 1

            brackets.append(a)

        a += step_size
        b += step_size

    for i in range(int(len(brackets) / 2)):
        j = 2*i + brackets_start
        a = brackets[j]
        b = brackets[j + 1]
        root = find_root(a, b, df)
        roots.append(root)
    
    return roots

# calculates the average of a list of numbers
def mean(mylist):
    list_lenght = len(mylist)
    my_sum = 0
    for i in range(list_lenght):
        my_sum += mylist[i]
    return my_sum / list_lenght

# calculates the standard deviation of a list of numbers
def standard_dev(mylist):
    list_length = len(mylist)
    my_sum = 0
    average = mean(mylist)
    for i in range(list_length):
        my_sum += (mylist[i] - average)**2

    return np.sqrt(my_sum/list_length)


""" SECOND HALF: OBTAINING RESULTS """


dt = .01 # resolution
t_start = 0 # this will allways be 0
t_end = 200 # the maximum value of time 

t = np.arange(t_start, t_end, dt)
state_0 = [10, 10, 14] # initial conditions
y = odeint(system, state_0, t) # y is a list with elemets that are list with 3 entries containing u, Te, Tw for each time step

# this corresponds to the functions u, Te, Tw which are function of time
u = y[:,0]
Te = y[:,1]
Tw = y[:,2]

# first plots

plt.plot(t, u)
plt.title("Current velocity against time ({} years)".format(t_end))
plt.xlabel("Time t (years)")
plt.ylabel("Current Velocity u ") # 10^3km / year  units?
plt.ylim((-400,400))
plt.xlim((0, t_end))
plt.grid(axis = 'y')
#plt.show()

"""
plt.plot(t, Te - Tw)
plt.title("Difference in Temp against time ({} years)".format(t_end))
plt.xlabel("Time t (years)")
plt.ylabel("T_e - T_w ")
plt.ylim((-30,30))
plt.show()
"""

du = diff(u, t, dt) # calculating derivative of u
# this will plot the derivative of u
#plt.plot(t[1:-1], du, "--")

roots = find_all_maxima(u, du, t_start, t_end, .1, 100) # finding all the 

# code for single root finder
"""root = find_root(83, 84, du)
print(root)
print(func_u(root))
plt.scatter(root, func_u(rot))
"""

# number of El-Nino events in the time run
num_of_elNinos = len(roots)
print("The number of ENSO events in {} years is: {}".format(t_end, num_of_elNinos))

# calculating the periods of time that elapses between the El-Nino events
times_between_ENSO = [] # periods of time
for i in range(num_of_elNinos - 1):
    times_between_ENSO.append(roots[i+1] - roots[i]) # roots is sorted so dont need abs() function

# finding the mean, and standard deviations (igonring the first 10 ENSO events)
average_T = mean(times_between_ENSO[10:])
dev_T = standard_dev(times_between_ENSO[10:])
print("The mean time between ENSO events (ignoring the first 10) is: {} years".format(average_T))
print("The standard deviation of ENSO events (ignoring the first 10) is: {}".format(dev_T))

# this will mark all the maxima in the list of roots on the graph
for i in range(len(roots)):
    plt.scatter(roots[i], make_func(u, roots[i]))

# histogram plot make nicer
"""plt.hist(times_between_ENSO, bins= 30)
plt.ylabel("Number of ENSO events")
plt.xlabel("Time beteen ENSO events")
plt.show()"""

# ADRISNANANA
# plot of the fractal 8 figure thing
"""plt.plot(u, Te - Tw)
plt.title("Difference in Temp against current velocity".format(t_end))
plt.xlabel("current velocity (1000 km / years)")
plt.ylabel("T_e - T_w ")
plt.ylim((-30,30))
plt.show()"""

plt.show()

"""# plot of ti against ti+10, ignoring the first 10 events
plt.plot(times_between_ENSO[10:-10], times_between_ENSO[20:])
plt.title("Ti vs T(i+10)")
plt.xlabel("T(i+10)")
plt.ylabel("Ti")
plt.show()
# correlation = stairs
# if uncorrelated, graph would have no distinguishable pattern"""

"""ENSO_months = []
for i in range(len(mod_roots)):
    ENSO_months.append(mod_roots[i]%1)
    
    
  # histogram plot make nicer
plt.hist(ENSO_months, bins= 12)
plt.ylabel("ENSO event")
plt.xlabel("Months")
plt.show()


mod_roots = find_all_maxima(mod_u, du, t_start, t_end, .1, 100)
# code for single root finder
    """
