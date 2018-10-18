#Створення прикладних програм на мові Python. Лабораторна робота №8.2. Гураль Руслана. FI-9119

import math

print('''' Створення прикладних програм на мові Python. Лабораторна робота №8.2.
  Гураль Руслана. FI-9119''')



def func(n, *args, **kwargs):
    if len(args) < n:
        for i in range(len(args), n):
            args[i] = 0
    if len(kwargs) < n:
        for i in range(len(kwargs), n):
            kwargs[i] = 0
    x = 0
    y = 0
    result = 0

    for value in kwargs.values():
        y += float(value)**2;

    for i in range(0, len(args)):
        x += args[i]**2

    result = x * y

    print('F(x_1,x_2,...,x_n,y_1,y_2,...y_n) = ', result)

func(1,2,3, a = 1, b = 2, c = 3)