from deap.tools import Statistics
from numpy import mean, max, median, std, linalg, min

def get_fitnesses(population):
    return list(map(lambda ind: ind.fitness.values[0], population))

class RunStats(Statistics):
    def __init__(self):
        super().__init__()
        self.register('mean', self.mean)
        self.register('max', self.max)
        self.register('median', self.median)
        self.register('min', self.min)
        self.register('std', self.std)
        self.register('diversity', self.diveristy)

    def mean(self, population):
        fitnesses = get_fitnesses(population)
        return mean(fitnesses)

    def max(self, population):
        return max(get_fitnesses(population))

    def median(self, population):
        return median(get_fitnesses(population))

    def std(self, population):
        return std(get_fitnesses(population))

    def diveristy(self, population):
        distances = []
        for ind in population:
            inds_distances = [linalg.norm(ind - other) for other in population if other is not ind]
            distances.append(mean(inds_distances))
        return mean(distances)

    def min(self, population):
        return min(get_fitnesses(population))