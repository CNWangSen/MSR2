import numpy

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, dim_obs):
        self.n = 0
        self.mean = [0 for i in range(dim_obs)]
        self.S = [0 for i in range(dim_obs)]
        self.std = [0 for i in range(dim_obs)]

    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            self.old_mean = self.mean.copy()
            self.mean = [self.old_mean[i]+(x[i]-self.old_mean[i])/self.n for i in range(len(x))]#old_mean + (x - old_mean) / self.n
            self.S = [self.S[i]+(x[i]-self.old_mean[i])*(x[i]-self.mean[i]) for i in range(len(x))]#self.S + (x - old_mean) * (x - self.mean)
            self.std = [(self.S[i]/self.n)**0.5 for i in range(len(x))]#numpy.sqrt(self.S / self.n )
