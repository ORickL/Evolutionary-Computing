from itertools import repeat
from scipy.stats import levy
import random

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


def mutLevy(individual, *, mu=.5, c=.5, indpb=.05):
    """
    This function defines a custom mutation function not found in deap which follows a mutation by the
    Levy distribution
    """
    size = len(individual)
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
    if not isinstance(c, Sequence):
        c = repeat(c, size)
    elif len(c) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(c), size))

    for i, m, cs in zip(range(size), mu, c):
        r = random.random()
        if r < indpb:
            sign = -1 if r < indpb / 2. else 1
            lv = levy(m, cs)
            individual[i] += sign * lv.rvs()

    return individual,
