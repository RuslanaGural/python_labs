import numpy as np

print(np.__version__)
np.show_config()

print(' ')
# 3
z = np.zeros(10)
print(z)

print(' ')
# 4
z = np.ones(10)
print(z)

print(' ')
# 5
z = np.full(10, 2.5)
print(z)

# 6 python3 -c "import numpy; numpy.info(numpy.add) - документація про функцію add

print(' ')
# 7
z = np.zeros(10)
z[4] = 1
print(z)

print(' ')
# 8
z = np.arange(10, 50)
print(z)

print(' ')
# 9
z = np.arange(50)
print(z)
z = z[::-1]
print(z)

print(' ')
# 10
z = np.arange(9).reshape(3, 3)
print(z)

print(' ')
# 11
nz = np.nonzero([1, 2, 0, 0, 4, 0])
print(nz)

print(' ')
# 12
z = np.eye(3)
print(z)

print(' ')
# 13
z = np.random.random((3, 3, 3))
print(z)

print(' ')

# 14
z = np.random.random((10, 10))
zmin, zmax = z.min(), z.max()
print(z)
print('min = ', zmin, ' max = ', zmax)

print(' ')
# 15
z = np.random.random(30)
m = z.mean()
print(z)
print(m)

print(' ')
# 16
z= np.ones((10,10))
z[1:-1, 1:-1] = 0
print(z)

print(' ')
# 17
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
0.3 == 3 * 0.1

print(' ')
# 18
z = np.diag(np.arange(1, 5), k=-1)
print(z)

print(' ')
# 19
z = np.zeros((8, 8), dtype=int)
z[1::2, ::2] = 1
z[::2, 1::2] = 1
print(z)

print(' ')
# 20
print(np.unravel_index(100,(6,7,8)))

print(' ')
# 21
z = np.tile(np.array([[0, 1], [1, 0]]), (4,4))
print(z)

print(' ')
# 22
z = np.dot(np.ones((5, 3)), np.ones((3,2)))
print(z)

print(' ')
# 23
z = np.arange(11)
z[(3 < z) & (z <= 8)] *= -1
print(z)

print(' ')
# 24
z = np.zeros((5, 5))
z += np.arange(5)
print(z)

print(' ')
# 25
def generate():
    # for x in xrange(10):
    for x in range(10):
        yield x
z = np.fromiter(generate(),dtype = float,count = -1)
print(z)

print(' ')
# 26
z = np.linspace(0, 1, 12)[1:-1]
print(z)

print(' ')
# 27
z = np.random.random(10)
z.sort()
print(z)

print(' ')
# 28
A = np.random.randint(0, 2, 5)
B = np.random.randint(0, 2, 5)
equal = np.allclose(A,B)
print(equal)

print(' ')
# 29
z = np.zeros(10)
z.flags.writeable = False
# z[0] = 0
print(z)

print(' ')
# 30
z = np.random.random((10, 2))
x, y = z[:, 0], z[:, 1]
r = np.hypot(x, y)
t = np.arctan2(y, x)
print(r)
print(t)

print(' ')
# 31
z = np.random.random(10)
z[z.argmax()] = 0
print(z)

print(' ')
# 32
z = np.zeros((10, 10), [('x', float), ('y', float)])
z['x'], z['y'] = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
print(z)

print(' ')
# 33
x = np.arange(8)
y = x + 0.5
c = 1.0 / np.subtract.outer(x, y)
print(np.linalg.det(c))

print(' ')
# 34
for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)

print(' ')
# 35
np.set_printoptions(threshold=np.nan)
z = np.zeros((25, 25))
print(z)

print(' ')
# 36
z= np.arange(100)
v = np.random.uniform(0, 100)
index = (np.abs(z - v)).argmin()
print(z[index])

print(' ')
# 37
z = np.zeros(10, [('position', [('x', float, 1),
                                ('y', float, 1)]),
                  ('color',    [('r', float, 1),
                                ('g', float, 1),
                                ('b', float, 1)])])
print(z)

print(' ')
# 38
import scipy.spatial

z = np.random.random((10, 2))
d = scipy.spatial.distance.cdist(z, z)
print(d)

print(' ')
# 39
z = np.arange(10, dtype=np.int32)
z = z.astype(np.float32, copy=False)

print(' ')
# 40
# z = np.genfromtxt("missing.dat", delimiter=",") - для зчитування файлу

print(' ')
# 41
z = np.arange(9).reshape(3, 3)
for index, value in np.ndenumerate(z):
    print(index, value)
for index in np.ndindex(z.shape):
    print(index, z[index])

print(' ')
# 42
x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
d = np.hypot(x, y)
sigma, mu = 1.0, 0.0
g = np.exp(-((d - mu)**2 / (2.0 * sigma ** 2)))
print(g)

print(' ')
# 43
n = 10
p = 3
z = np.zeros((n,n))
np.put(z, np.random.choice(range(n*n), p, replace=False), 1)

print(' ')
# 44
x = np.random.rand(5, 10)
y = x - x.mean(axis=1, keepdims=True)

print(' ')
# 45
z = np.random.randint(0, 10, (3, 3))
n = 1 #нумерація з нуля
print(z)
print(z[z[:, n].argsort()])

print(' ')
# 46
z = np.random.randint(0, 3, (3, 10))
print((~z.any(axis=0)).any())

print(' ')
# 47
z = np.ones(10)
i = np.random.randint(0, len(z), 20)
z += np.bincount(i, minlength=len(z))
print(z)

print(' ')
# 48
w, h = 16, 16
i = np.random.randint(0, 2, (h, w, 3)).astype(np.ubyte)
f = i[..., 0] * 256 *256 + i[..., 1] * 256 + i[..., 2]
n = len(np.unique(f))
print(np.unique(i))

print(' ')
#49
A = np.random.randint (0, 10, (3, 4, 3, 4))
sum = A.reshape(A.shape[: 2] + (-1,)).sum(axis=-1)
print(sum)

print(' ')
#50
# np.diag(np.dot(A, B))

# np.sum(A * B.T, axis=1)
#
# np.eisum("ij,ji->", A, B)

print(' ')
#51
z = np.array([1, 2, 3, 4, 5])
nz = 3
z0 = np.zeros(len(z) + (len(z) - 1) * (nz))
z0[::nz + 1]=z
print(z0)

print(' ')
#52
A = np.arange(25).reshape(5, 5)
A[[0, 1]]=A[[1, 0]]
print(A)

print(' ')
#53
faces = np.random.randint(0, 10, (10, 3))
f = np.roll(faces.repeat(2, axis=1), -1, axis=1)
f = f.reshape(len(f) * 3, 2)
f = np.sort(f, axis=1)
g = f.view(dtype = [('p0', f.dtype), ('p1', f.dtype)])
g = np.unique(g)
print(g)

print(' ')
#54
C = np.bincount([1, 1, 2, 3, 4, 4, 6])
A = np.repeat(np.arange(len(C)), C)
print(A)

print(' ')
#55
def moving_average(a, n = 3):
    ret = np.cumsum(a, dtype =  float)
    ret[n:] = ret[n:] - ret[: -n]
    return ret [n-1 :] / n
print(moving_average(20), 3)

print(' ')
#56
from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
z = rolling(np.arange(10), 3)
print(z)

print(' ')
#57
z = np.random.randint(0, 2, 10)
print(z)
np.logical_not(z, out=z)
print(z)

z = np.random.uniform(-1.0,1.0,100)
print(z)
np.negative(z, out=z)
print(z)

print(' ')
#58
def distance(p0, p1, p):
    t = p1 - p0
    l = (t ** 2).sum(axis=1)
    u = -((p0[:, 0] - p[..., 0]) * t[:, 0] + (p0[:, 1] - p[..., 1]) * t[:, 1]) / l
    u = u.reshape(len(u), 1)
    d = p0 + u * t - p
    return np.sqrt((d ** 2).sum(axis=1))
p0= np.random.uniform(-10, 10, (10, 2))
p1= np.random.uniform(-10, 10, (10, 2))
p= np.random.uniform(-10, 10, (1, 2))
print(distance(p0, p1, p))

print(' ')
#59
z = np.random.randint(0, 10, (10, 10))
shape = (5, 5)
fill = 0
position = (1, 1)

r = np.ones(shape, dtype = z.dtype) * fill
p = np.array(list(position)).astype(int)
rs = np.array(list(r.shape)).astype(int)
zs = np.array(list(z.shape)).astype(int)

r_start = np.zeros((len(shape),)).astype(int)
r_stop = np.array(list(shape)).astype(int)
z_start = (p - rs // 2)
z_stop = (p + rs // 2) + rs % 2

r_start = (r_start - np.minimum(z_start, 0)).tolist()
z_start = (np.maximum(z_start, 0)).tolist()
r_stop = np.maximum(r_start, (r_stop - np.maximum(z_stop - zs, 0))).tolist()
z_stop =(np.minimum(z_stop, zs)).tolist()

R = [slice(start, stop) for start, stop in zip(r_start, r_stop)]
Z = [slice(start, stop) for start, stop in zip(z_start, z_stop)]
r[R]=z[Z]
print(z)
print(r)

print(' ')
#60
z =  np.random.uniform(0, 1, (10, 10))
rank = np.linalg.matrix_rank(z)
print(rank)

print(' ')
#61
z = np.random.randint(0, 10, 50)
print(np.bincount(z).argmax())

print(' ')
#62
z = np.random.randint(0, 5, (10, 10))
n = 3
i = 1 + (z.shape[0] - n)
j = 1 + (z.shape[1] - n)
c = stride_tricks.as_strided(z, shape=(i, j, n, n), strides=z.strides + z.strides)
print(c)

print(' ')
#63
# class Symetric(np.ndarray):
#     def __setitem__(self, (i,j), value):
#         super(Symetric, self).__setitem__((i, j), value)
#         super(Symetric, self).__setitem__((i, j), value)
#
#
# def symetric(z):
#     return np.asarray(z + z.T - np.diag(z.diagonal())).view(Symetric)
#
#
# S = symetric(np.random.randint(0, 10, (5, 5)))
# S[2, 3] = 42
# print(S)

print(' ')
# 64
p, n = 10, 20
M = np.ones((p, n, n))
V = np.ones((p, n, 1))
S = np.tensordot(M, V, axes=[[0, 2],[0, 1]])
print(S)

print(' ')
# 65
Z = np.ones((16, 16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(' ')
# 66
def iterate(Z):
    N = (Z[0:-2, 0:-2] + Z[0:-2, 1:-1] + Z[0:-2, 2:] +
         Z[1:-1, 0:-2] + Z[1:-1, 2:]   + Z[2:, 0:-2] +
         Z[2:  , 1:-1] + Z[2:  , 2:])

    birth = (N == 3) & (Z[1:-1, 1:-1] == 0)
    survive = ((N == 2) | (N == 3)) & (Z[1:-1, 1:-1] == 1)
    Z[...] = 0
    Z[1:-1, 1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0, 2, (50, 50))
for i in range(100):
    print(Z)
    Z = iterate(Z)


print(' ')
# 67
z = np.arange(10000)
np.random.shuffle(Z)
n = 5
print(Z[np.argpartition(-Z, n)[:n]])


print(' ')
# 68
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = map(len, arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print(cartesian(([1, 2, 3], [4, 5], [6, 7])))


print(' ')
# 69
A = np.random.randint(0, 5, (8, 3))
B = np.random.randint(0, 5, (2, 2))
C = (A[..., np.newaxis, np.newaxis] == B)
rows = (C.sum(axis=(1, 2, 3)) >= B.shape[1]).nonzero()[0]
print(rows)


print(' ')
# 70
# Z = np.randon.randint(0, 5, (10, 3))
# E = np.logical_and.reduce(Z[:, 1:] == Z[:, :-1], axis=1)
# U = Z[~E]
# print(Z)
# print(U)

print(' ')
# 71
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))

print(' ')
# 72
Z = np.random.randint(0, 2, (6, 3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)

print(' ')
# 73
# np.einsum('i->', A)
# np.einsum('i,j->i', A, B)
# np.einsum('i,i', A, B)
# np.einsum('i,j', A, B)







