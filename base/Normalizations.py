import numpy as np

class MaxMinusMin(object):
    """ ``Max minus min'' normalization. The normalization is done by dividing
        the time series by the maximum minus minimum.
    """
    name = 'max_minus_min'

    def __init__(self, params):
        """ Create MaxMinusMin normalization object.
            Args:
                params (tuple): parameters of the normalization. For
                    MaxMinusMin, the params are `delta' and `denominator'.
        """

        self.params = params
        self.delta, self.denominator = params

    @staticmethod
    def from_dataseries(u, shift=False):
        if shift:
            delta = u.mean(axis=0)
        else:
            delta = np.zeros(u.shape[1])

        denominator = u.max(axis=0) - u.min(axis=0)

        return MaxMinusMin((delta, denominator))

    def normalize(self, v):
        return (v-self.delta)/self.denominator

    def denormalize(self, v):
        return self.denominator*v + self.delta


NORMALIZATIONS = {MaxMinusMin.name: MaxMinusMin}
