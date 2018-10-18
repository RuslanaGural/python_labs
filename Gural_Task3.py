#Створення прикладних програм на мові Python. Лабораторна робота №3. Гураль Руслана. FI-9119

import math

print('''' Створення прикладних програм на мові Python. Лабораторна робота №3
  Гураль Руслана. FI-9119''')

x = float(input('x = '))
y = float(input('y = '))

z = ( 5 * x ) / ( math.pow( x, 3 ) + math.pow( y, 3 ) ) - math.cos( 3 * x / y ) / math.sin( 3 * x / y )

if math.isfinite(z):
    print('z = ', z)
else:
    print('значення змінних виходять за область визначення функції')
