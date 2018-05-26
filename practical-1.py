import torch, time
from torch import Tensor

def ex1():
    print('# ex1 #')
    matrix = Tensor(13, 13).fill_(1)
    
    # Fill 2 to horizontal lines
    matrix.narrow(0, 1, 1).fill_(2)
    matrix.narrow(0, 6, 1).fill_(2)
    matrix.narrow(0, 11, 1).fill_(2)

    # Fill 2 to vertical lines
    matrix.narrow(1, 1, 1).fill_(2)    
    matrix.narrow(1, 6, 1).fill_(2)
    matrix.narrow(1, 11, 1).fill_(2)

    # Fill 3 to blocks
    matrix.narrow(0, 3, 2).narrow(1, 3, 2).fill_(3)
    matrix.narrow(0, 3, 2).narrow(1, 8, 2).fill_(3)
    matrix.narrow(0, 8, 2).narrow(1, 3, 2).fill_(3)
    matrix.narrow(0, 8, 2).narrow(1, 8, 2).fill_(3)
    
    print(matrix)


def ex2():
    print('# ex2 #')

    matrix = Tensor(20, 20).normal_()
    diag = torch.diag(torch.arange(1, matrix.size(0)+1))
    q = matrix.mm(diag).mm(matrix.inverse())
    v, _ = q.eig()
    print('Eigenvalues', v.narrow(1, 0, 1).squeeze().sort()[0])


def ex3():
    print('# ex3 #')
    dimension = 5000
    a = torch.cuda.FloatTensor(dimension, dimension).normal_()
    b = torch.cuda.FloatTensor(dimension, dimension).normal_()

    time1 = time.perf_counter()
    c = torch.mm(a, b)
    time2 = time.perf_counter()

    print('Throughput {0:.2f} Tflop/s'.format((dimension**3)/(time2 - time1)/(10**12)))


def ex4():
    print('# ex4 #')
    def mul_row(matrix):
        for i in range(0, matrix.size(0)):
            for j in range(0, matrix.size(1)):
                matrix[i][j] *= (i + 1)
        return matrix

    def mul_row_fast(matrix):
        factor = torch.arange(1, matrix.size(0) + 1).view(matrix.size(0), 1)
        matrix = torch.mul(matrix, factor)
        return matrix

    matrix = Tensor(10000, 400).normal_()
    time1 = time.perf_counter()
    mul_row(matrix)
    time2 = time.perf_counter()
    mul_row_fast(matrix)
    time3 = time.perf_counter()
    
    print('mul_row: {0:.2f}'.format(time2 - time1))
    print('mul_row_fast: {0:.2f}'.format(time3 - time2))


def main():
    ex1()
    ex2()
    ex3()
    ex4()


if __name__ == '__main__':
    main()