

def twoints(v):
    return (int(v[0]), int(v[1]))


def loglevel(v):
    loglevels = ("debug", "info", "warning", "error")
    if v not in loglevels:
        raise ValueError("loglevel must be one of %r" % loglevels)
    return v


AVAILABLE_SETTINGS = (
    # (name, type, default)
    ("backend", str, "numpy"),
    ("num_proc", twoints, (1, 1)),
    ("profile_mode", bool, False),
    ("loglevel", loglevel, "info"),
    ("logfile", str, None),
)


class RuntimeSettings(object):
    def __init__(self):
        self.__locked__ = False
        self.__setting_types__ = {}

        for setting, typ, default in AVAILABLE_SETTINGS:
            setattr(self, setting, default)
            self.__setting_types__[setting] = typ

        self.__settings__ = set(self.__setting_types__.keys())
        self.__locked__ = True

    def __setattr__(self, attr, val):
        if attr == "__locked__" or not self.__locked__:
            return super(RuntimeSettings, self).__setattr__(attr, val)

        # prevent adding new settings
        if attr not in self.__settings__:
            raise AttributeError("Unknown runtime setting %s" % attr)

        # check type
        stype = self.__setting_types__.get(attr)
        if stype is not None:
            # types are either callables or proper types
            if callable(stype):
                val = stype(val)
            elif not isinstance(val, stype):
                raise TypeError("invalid type for setting %s" % setting)

        return super(RuntimeSettings, self).__setattr__(attr, val)

    def __repr__(self):
        setval = ",".join(
            "%s=%s" % (key, getattr(self, key)) for key in self.__settings__
        )
        return "{clsname}({setval}".format(
            clsname=self.__class__.__name__,
            setval=setval
        )
