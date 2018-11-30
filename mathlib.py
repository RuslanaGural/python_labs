from numpy import *
from math import *
import matplotlib.pyplot as plt

def f(x):
    return sin(10*x) * sin(3*x) / x**2

x=linspace(0,4,100)

plt.plot(x,f(x),label='(sin(10*x)*sin(3*x)) / x**2')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()

plt.savefig('sinus.png',dpi=200)