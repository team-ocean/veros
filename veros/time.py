from . import veros_method

YEAR_LENGTH = 360.
X_TO_SECONDS = {
    "seconds": 1.,
    "minutes": 60.,
    "hours": 60. * 60.,
    "days": 24. * 60. * 60.,
    "years": YEAR_LENGTH * 60. * 60.
}
SECONDS_TO_X = {key: 1. / val for key, val in X_TO_SECONDS.items()}

@veros_method
def current_time(veros, unit="seconds"):
    time_in_seconds = (veros.itt - 1) * veros.dt_tracer
    return convert_time(veros, time_in_seconds, "seconds", unit)

@veros_method
def convert_time(veros, time_value, in_unit, out_unit):
    return time_value * X_TO_SECONDS[in_unit] * SECONDS_TO_X[out_unit]
