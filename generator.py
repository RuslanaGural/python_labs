#from math import *

import math


def genFunc (x, eps):
    if math.fabs(x) < 1:
        count = 0
        print('x_', count, ' = ', x)

        while True:
            yield x
            if math.fabs(x) > eps:
                count = count + 1
                x = math.pow(-1, count-1) * math.pow(x, count) / count;
                print('x_', count, ' = ', x)
            else:
                break
    else:
        print("Не вірно введені дані. |x| < 1.")

result = 0

for i in genFunc(0.9, 0.01):
    result = result + i

print('Результат функції', result)