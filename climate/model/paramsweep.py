import numpy
import climate.io.wrapper as wrapper
from multiprocessing import Process

class dist:
    normal, random = range(2)

def paramsweep(callback, samples, **kwargs):
    """ generate random samples based on values """
    vals = {}
    for var in kwargs:
        if isinstance(kwargs[var], tuple):
            (minVal, maxVal, distribution) = kwargs[var]
            if distribution == dist.normal:
                vals[var] = normal(samples, minVal, maxVal)
            elif distribution == dist.random:
                if isinstance(minVal, float) or isinstance(maxVal, float):
                    vals[var] = randomFloat(samples, minVal, maxVal)
                else:
                    vals[var] = randomInt(samples, minVal, maxVal)
        else:
            vals[var] = kwargs[var]
    ps = []
    for i in xrange(samples):
        callVals = {}
        for var in vals:
            if isinstance(vals[var], numpy.ndarray):
                callVals[var] = vals[var][i]
            else:
                callVals[var] = vals[var]
        p = Process(target=callback, kwargs=callVals)
        ps.append(p)
        p.start()
    [p.join() for p in ps]

def normal(samples, minVal, maxVal):
    # Normal distribution from 0 to 2
    distribution = numpy.random.standard_normal(samples) + 1
    # From 0 to (maxVal - minVal)
    distribution *= (maxVal - minVal) / 2.
    # From minVal to maxVal
    distribution += minVal
    return distribution

def randomFloat(samples, minVal, maxVal):
    return numpy.random.uniform(minVal, maxVal, samples)

def randomInt(samples, minVal, maxVal):
    return numpy.random.randint(minVal, maxVal, samples)
