import numpy

class dist:
    normal, random = range(2)

# Monte carlo simulation data
def montecarlo(callback, samples, **kwargs):
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
    for i in xrange(samples):
        callVals = {}
        for var in vals:
            if isinstance(vals[var], numpy.ndarray):
                callVals[var] = vals[var][i]
            else:
                callVals[var] = vals[var]
        callback(**callVals)

def normal(samples, minVal, maxVal):
    mean = (maxVal + minVal) / 2.
    deviation = (mean - minVal) / 3.
    return numpy.random.normal(mean, deviation, samples)

def randomFloat(samples, minVal, maxVal):
    return numpy.random.uniform(minVal, maxVal, samples)

def randomInt(samples, minVal, maxVal):
    return numpy.random.randint(minVal, maxVal, samples)
