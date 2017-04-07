from collections import OrderedDict
import h5py

class VerosDiagnostic(object):
    extra_variables = None
    extra_conditional_variables = None
    sampling_frequency = float("inf")
    output_frequency = float("inf")
    outfile = None

    @property
    def is_active(self):
        return self.sampling_frequency or self.output_frequency

    def get_variable_array(self, veros, var):
        if var in self.extra_variables.keys():
            return getattr(self, var)
        elif var in veros.variables.keys():
            return getattr(veros, var)
        else:
            raise AttributeError("attribute {} not found".format(var))

    def read_restart(self, veros):
        pass

    def write_restart(self, veros):
        pass

    def _not_implemented(self):
        raise NotImplementedError("must be implemented by subclass")

    initialize = _not_implemented
    diagnose = _not_implemented
    output = _not_implemented
