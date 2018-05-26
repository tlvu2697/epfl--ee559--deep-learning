#!/usr/bin/env python

import torch
from torch import Tensor

import dlc_practical_prologue as prologue

######################################################################
# Classify

def nearest_classification(train_input, train_target, x):
    dist = (train_input - x).pow(2).sum(1).view(-1)
    _, n = torch.min(dist, 0)
    return train_target[n[0]]

######################################################################
# Compute the mean and PCA basis

def PCA(x):
    mean = x.mean(0)
    b = x - mean
    Sigma = b.t().mm(b)
    eigen_values, eigen_vectors = Sigma.eig(True)
    right_order = eigen_values[:,0].abs().sort(0, True)[1]
    eigen_vectors = eigen_vectors.t()[right_order]
    return mean, eigen_vectors

######################################################################

def compute_nb_errors(train_input, train_target,
                      test_input, test_target,
                      mean = None, proj = None):

    if mean is not None:
        train_input = train_input - mean
        test_input = test_input - mean

    if proj is not None:
        train_input = train_input.mm(proj.t())
        test_input = test_input.mm(proj.t())

    nb_errors = 0

    # With loop, but I prefer clearer code when counting errors
    for n in range(0, test_input.size(0)):
        if test_target[n] != nearest_classification(train_input, train_target, test_input[n]):
            nb_errors = nb_errors + 1

    return nb_errors

######################################################################

for c in [ False, True ]:

    train_input, train_target, test_input, test_target = prologue.load_data(cifar=c)

    nb_errors = compute_nb_errors(train_input, train_target, test_input, test_target)
    print('Baseline nb_errors {:d} error {:.02f}%'.format(nb_errors, 100 * nb_errors / test_input.size(0)))

    ##

    basis = train_input.new(100, train_input.size(1)).normal_()

    nb_errors = compute_nb_errors(train_input, train_target, test_input, test_target, None, basis)
    print('Random {:d}d nb_errors {:d} error {:.02f}%'.format(basis.size(0), nb_errors, 100 * nb_errors / test_input.size(0)))

    ##

    mean, basis = PCA(train_input)

    for d in [ 100, 50, 10, 3 ]:
        basis = basis.narrow(0, 0, d)
        nb_errors = compute_nb_errors(train_input, train_target, test_input, test_target, mean, basis)
        print('PCA {:d}d nb_errors {:d} error {:.02f}%'.format(d, nb_errors, 100 * nb_errors / test_input.size(0)))

######################################################################
