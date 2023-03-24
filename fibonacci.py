a = 20

def fibonacci(n):
    """
    f = fibonnaci(n): returns a list of the n first fibonacci numbers
    """
    f = [1] * n
    for i in range(2,n):
        f[i] = f[i-2] + f[i-1]
    return f

def myfib(n):
    list = [1, 1]
    for i in range(n-2):
        entry = list[i] + list[i+1]
        list.append(entry)
    return list

print(fibonacci(a))
print(myfib(a))