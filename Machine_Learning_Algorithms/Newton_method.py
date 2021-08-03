# Newton method exmaples writen by Yurong Chen
import math

# Case 1: f(x) = 2*x - 3*x**2 + x**3
def func(x):
    return 2*x - 3*x**2 + x**3

def differetiate(x):
    h = 1e-5
    return (func(x+h) - func(x)) / h

def caluate(x):
    return - (func(x) / differetiate(x)) + x

x = 0
res = [10,10]
while abs(x - res[-2]) >= 0.01:
    temp = caluate(x)
    res.append(temp)
    x = temp
    print(x)
print(res)