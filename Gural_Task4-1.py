#Створення прикладних програм на мові Python. Лабораторна робота № 4.1. Гураль Руслана. FI-9119

import math

print('''' Створення прикладних програм на мові Python. Лабораторна робота №4.1
  Гураль Руслана. FI-9119''')

n = 1
s = 0
a = float(math.factorial(n)/n**n)
while a >= 10**(-4):
    s += a
    n += 1
    a = float(math.factorial(n) / n**n)
print('Результат: s = ', s)
print('Результат: a_n = ',  round(a,4))
print('Результат кількість ітерацій:  = ', n)
