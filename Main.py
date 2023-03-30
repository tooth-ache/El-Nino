import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# making the vectorised differential equations
def f(state, t):
    u, T_e, T_w = state

    # constants
    A = 1.0
    B = 663.0
    C = 3.0
    T_star = 12.0
    u_star = -14.2
    delta_x = 7.5

    # setting up the 3 differential equations
    du = B / delta_x * (T_e - T_w) - C * (u - u_star) #  = du/dt
    dTe = u * T_w / (2 * delta_x) - A * (T_e - T_star) #  = dTe/dt
    dTw = -u * T_e / (2 * delta_x) - A * (T_w - T_star) #  = dTw/dt

    return du, dTe, dTw

dt = .0001 # resolution

t_start = 0 # this will allways be 0
t_want = 200 # the end value we care about 
t_end =  2 * t_want # this will be for how many years this will run so that we can find all the roots 

t = np.arange(t_start, t_end, dt)
state_0 = [10, 10, 14] # initial conditions
y = odeint(f, state_0, t) # y is a list with elemets that are list with 3 entries containing u, Te, Tw for each time step

# this corresponds to the functions u, Te, Tw which are function of time
u = y[:,0]
Te = y[:,1]
Tw = y[:,2]

# first plots

plt.plot(t, u)
plt.title("Current velocity against time ({} years)".format(t_want))
plt.xlabel("Time t (years)")
plt.ylabel("Current Velocity u ") # 10^3km / year  units?
plt.ylim((-400,400))
plt.xlim((0, t_want))
plt.grid(axis = 'y')
#plt.show()

"""
plt.plot(t, Te - Tw)
plt.title("Difference in Temp against time ({} years)".format(t_want))
plt.xlabel("Time t (years)")
plt.ylabel("T_e - T_w ")
plt.ylim((-30,30))
plt.show()
"""

# function for taking the derivative of a function (symmetric derivative)
def diff(func, x, dx):
    dfs = np.zeros(len(x) - 2)
    for i in range(1, len(x) - 1):
        dfs[i-1] = (func[i + 1] - func[i - 1]) / (2 * dx)
    return dfs

# this will the numpy array into a something that work like a mathematical function
def make_func(list_func, x):
    if len(list_func) - 1 < int(round(x/dt)):
        return list_func[-1]
    else:
        return list_func[int(round(x/dt))]

# function for root finding (bisection method)
def find_root(a, b, func):
    x1 = 0
    x2 = 0
    x3 = 0
    if a * b > 0:
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

du = diff(u, t, dt)

def func_u(x):
    return make_func(u, x)

# idea: run the rootfinder of increments of .5 years in t to find roots, there will
# be spots in which there is no root but rootfinder should stop after a while
# also notice that it looks like all maxima we care about are above y = 100

# this function will run the bisection method on intervals of time to find all roots and the pick out the ones we want
# currently this function below consistently decides to miss out on 2 to 6 roots for some reason and they are allways in the end
# funny fix idea: run programm for more than you need and then just look at the values you care about
def iterate_find_root(f, df ,start_step, end_step, increment, min_y, num_iterations):
    roots = []
    for j in range(num_iterations):
        a = start_step + (j*increment)/num_iterations
        b = a + increment
        for i in range(int(round(end_step / increment))):
            root = find_root(a, b, df)
            a += increment
            if type(root) == float:
                if make_func(f, root) > min_y:
                    roots.append(root)

    # removes entries in roots which are very close to eachother in and makes a new set
    roots.sort()
    good_roots = roots
    if num_iterations > 1:
        for i, root_1 in enumerate(roots):
            for root_2 in roots[i+1:]:
                if root_2 - root_1 < .1:
                    good_roots.remove(root_2)

    return good_roots



# piece of shit doesnt work fucking numpy arrays are not lists and wont turn into lists.
def find_maxima(list_func, increment, min_y):
    real_list_func = []
    for i in range(len(list_func)):
        real_list_func.append(float(list_func[i]))
    maxima = []
    step = int(increment / dt)
    a = 0
    b = a + step
    for i in range(int(len(list_func) / step)):
        small_interval = real_list_func[a:b]
        max_val = max(small_interval)
        max_val_index = small_interval.index(max_val)
        a += step
        if max_val > min_y:
            maxima.append(t[max_val_index])
            
    return maxima


roots = iterate_find_root(u, du, t_start, t_end, .1 , 100, 3)
#roots = find_maxima(u, .5, 100)
roots.sort()

roots_wanted = [] # these roots are the ones we care about, which are from 0 to t_want
for i in range(len(roots)):
    if roots[i] <= t_want:
        roots_wanted.append(roots[i])

for i in range(len(roots_wanted)):
    plt.scatter(roots_wanted[i], func_u(roots_wanted[i]))

# code for single root finder
"""root = find_root(83, 84, du)
print(root)
print(func_u(root))
plt.scatter(root, func_u(rot))
"""

# number of El-Nino events in the time run
num_of_elNinos = len(roots_wanted)
print("The number of ENSO events in {} years is: {}".format(t_want, num_of_elNinos))

# calculating the periods of time that elapses between the El-Nino events
times_between_ENSO = [] # periods of time
for i in range(num_of_elNinos - 1):
    times_between_ENSO.append(roots_wanted[i+1] - roots_wanted[i]) # roots is sorted so dont need abs() function

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

# to ignore the first 10 events repace times_between_ENSO with times_between_ENSO[11:]
average_T = mean(times_between_ENSO)
dev_T = standard_dev(times_between_ENSO)
print("The mean time between ENSO events is: {} years".format(average_T))
print("The standard deviation of ENSO events is: {}".format(dev_T))

# this will plot the derivative of u
#plt.plot(t[1:-1], du, "--")
plt.show()

# histgram plot make nicer
"""
plt.hist(times_between_ENSO, bins= 30)
plt.ylabel("Number of ENSO events")
plt.xlabel("Time beteen ENSO events")
plt.show()
"""
# ADRISNANANA
# plot of the fractal 8 figure thing
plt.plot(u, Te - Tw)
plt.title("Difference in Temp against current velocity".format(t_want))
plt.xlabel("current velocity (1000 km / years)")
plt.ylabel("T_e - T_w ")
plt.ylim((-30,30))
plt.show()
