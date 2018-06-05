#!/usr/bin/env python

import torch, time

from torch import Tensor

######################################################################
print('# 1 #')

m = Tensor(13, 13).fill_(1)

m.narrow(0, 1, 1).fill_(2)
m.narrow(1, 1, 1).fill_(2)
m.narrow(0, 6, 1).fill_(2)
m.narrow(1, 6, 1).fill_(2)
m.narrow(0, 11, 1).fill_(2)
m.narrow(1, 11, 1).fill_(2)

m.narrow(0, 3, 2).narrow(1, 3, 2).fill_(3)
m.narrow(0, 3, 2).narrow(1, 8, 2).fill_(3)
m.narrow(0, 8, 2).narrow(1, 3, 2).fill_(3)
m.narrow(0, 8, 2).narrow(1, 8, 2).fill_(3)

print(m)

######################################################################
# numpy style

m = Tensor(13, 13).fill_(1)

m[3:5,3:5] = 3
m[8:10,3:5] = 3
m[3:5,8:10] = 3
m[8:10,8:10] = 3

m[:,1::5] = 2
m[1::5,:] = 2

print(m)

######################################################################
print('# 2 #')

m = Tensor(20, 20).normal_()
d = torch.diag(torch.arange(1, m.size(0)+1))
q = m.mm(d).mm(m.inverse())
v, _ = q.eig()
print('Eigenvalues', v.narrow(1, 0, 1).squeeze().sort()[0])

######################################################################
print('# 3 #')

d = 5000
a = Tensor(d, d).normal_()
b = Tensor(d, d).normal_()

time1 = time.perf_counter()
c = torch.mm(a, b)
time2 = time.perf_counter()

print('Throughput {:e} flop/s'.format((d * d * d)/(time2 - time1)))

######################################################################
print('# 4 #')

def mul_row(m):
    r = torch.Tensor(m.size())
    for i in range(0, m.size(0)):
        for j in range(0, m.size(1)):
            r[i, j] = m[i, j] * (i+1)
    return r

def mul_row_fast(m):
    d = m.size(0)
    c = torch.arange(1, d + 1).view(d, 1).expand_as(m)

    return m.mul(c)

m = Tensor(10000, 400).normal_(5.0)

time1 = time.perf_counter()
a = mul_row(m)
time2 = time.perf_counter()
b = mul_row_fast(m)
time3 = time.perf_counter()

print('Speed ratio', (time2 - time1) / (time3 - time2))

print('Sanity check: error is ', torch.norm(a - b))

######################################################################
