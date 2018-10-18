#Створення прикладних програм на мові Python. Лабораторна робота №5. Гураль Руслана. FI-9119

import math

print('''' Створення прикладних програм на мові Python. Лабораторна робота №5
  Гураль Руслана. FI-9119''')

i = 0
result = 0

def fibonachi(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return ( fibonachi(n-1) + fibonachi(n-2) )


while fibonachi(i) <= 1000:
    result += fibonachi(i)
    i += 1

print('SumFibonachi = ', result)


