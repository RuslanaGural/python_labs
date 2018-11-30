def testFunc (x):
    """ Обчислюємо суму аргумента в четвертій степені і чотири в степені, рівному аргументу.

    >>> testFunc (1)
    5.0

    >>> testFunc (2)
    32.0

    >>> testFunc (4)
    512.0


    """
    return ( x ** 4. + 4 ** x )

import doctest

doctest.testmod()
