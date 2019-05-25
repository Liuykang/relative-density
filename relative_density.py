import numpy as np

"""
function performance：
Relative Density (RD) method implementation
"""


def get_k_dis_sum(k, data, dat):
    """get the density of our k neighbors."""
    dataSetSize = len(dat)
    diffMat = np.tile(data, (dataSetSize, 1)) - dat  # tile :construct array by repeating inX dataSetSize times
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5  # get distance
    sortedDist = np.sort(distances)
    k_distance_sum = sortedDist[:k].sum()
    return k_distance_sum



def relative_density(train, k=9, d=1, verbose=False):
    """
    Relative density method
    :param train: training set, Y is part of {+1,-1}
    :param k: K neighbor number
    :param d: threshold
    :param verbose:
    :return: clean training set
    """
    relative_density = []
    nega_train = train[train[:, 0] == -1, 1:]
    posi_train = train[train[:, 0] == 1, 1:]
    train_x = train[:, 1:]
    for i in range(len(train)):
        if verbose:
            print("The current line is %d" % (i + 1))
        k_distances_sum_1 = get_k_dis_sum(k, train_x[i], nega_train)
        k_distances_sum_2 = get_k_dis_sum(k, train_x[i], posi_train)

        if train[i, 0] == -1:
            relative_density.append(k_distances_sum_1 / k_distances_sum_2)
        else:
            relative_density.append(k_distances_sum_2 / k_distances_sum_1)

    relative_density = np.array(relative_density)
    new_train = train[relative_density <= d]
    return new_train
