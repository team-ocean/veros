import numpy

# Monte carlo simulation data
def montecarlo(callback, samples, **kwargs):
    """ generate random samples based on values """
    retVals = {}
    for var in kwargs:
        if isinstance(kwargs[var], tuple):
            (minVal, maxVal, dist) = kwargs[var]
            if dist == "normal":
                retVals[var] = normal(samples, minVal, maxVal)
            elif isinstance(minVal, float) or isinstance(maxVal, float):
                retVals[var] = randomFloat(samples, minVal, maxVal)
            else:
                retVals[var] = randomInt(samples, minVal, maxVal)
        else:
            retVals[var] = kwargs[var]
    callback(**retVals)

def normal(samples, mean, deviation):
    return numpy.random.normal(mean, deviation, samples)

def randomFloat(samples, minVal, maxVal):
    return numpy.random.uniform(minVal, maxVal, samples)

def randomInt(samples, minVal, maxVal):
    return numpy.random.randint(minVal, maxVal+1, samples)
