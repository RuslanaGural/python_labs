#Створення прикладних програм на мові Python. Лабораторна робота № 4.3. Гураль Руслана. FI-9119

import math

print('''' Створення прикладних програм на мові Python. Лабораторна робота №4.3
  Гураль Руслана. FI-9119''')

a = float(input( 'a = ') )
x = float( input( 'x = ') )

n = 0

while abs( x ** 2 - a) > 0.0001:
    x = 1 / 2 * ( x + a / x )
    n += 1

print('Корінь квадратний з а =  ', x)
print('Кількість ітерацій n = ', n)
