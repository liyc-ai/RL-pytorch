import numpy as np

# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/
# class RunningMeanStd(object):
#     def __init__(self, shape):
#         self._n = 0
#         self._M = np.zeros(shape)
#         self._S = np.zeros(shape)

#     def push(self, x):
#         x = np.asarray(x)
#         assert x.shape == self._M.shape
#         self._n += 1
#         if self._n == 1:
#             self._M[...] = x
#         else:
#             oldM = self._M.copy()
#             self._M[...] = oldM + (x - oldM) / self._n
#             self._S[...] = self._S + (x - oldM) * (x - self._M)

#     @property
#     def n(self):
#         return self._n

#     @property
#     def mean(self):
#         return self._M

#     @property
#     def var(self):
#         return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

#     @property
#     def std(self):
#         return np.sqrt(self.var)

#     @property
#     def shape(self):
#         return self._M.shape

# class ZFilter:
#     """
#     y = (x-mean)/std
#     using running estimates of mean,std
#     """

#     def __init__(self, shape, demean=True, destd=True, clip=10.0):
#         self.demean = demean
#         self.destd = destd
#         self.clip = clip

#         self.rs = RunningMeanStd(shape)

#     def __call__(self, x, update=True):
#         if update: self.rs.update(x)
#         if self.demean:
#             x = x - self.rs.mean
#         if self.destd:
#             x = x / (self.rs.std + 1e-8)
#         if self.clip:
#             x = np.clip(x, -self.clip, self.clip)
#         return x

class RunningMeanStd(object):
    """Calculates the running mean and std of a data stream.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(
        self,
        mean = 0.0,
        std = 1.0
    ) -> None:
        self.mean, self.var = mean, std
        self.count = 0

    def update(self, data_array: np.ndarray):
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = np.mean(data_array, axis=0), np.var(data_array, axis=0)
        batch_count = len(data_array)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count
        
class Normalizer:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self):
        self.rms = RunningMeanStd()
        
    def __call__(self, data_array: np.ndarray):
        self.rms.update(data_array)
        return (data_array-self.rms.mean)/(np.sqrt(self.rms.var)+1e-8)
    