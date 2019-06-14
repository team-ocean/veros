
YEAR_LENGTH = 360.
X_TO_SECONDS = {
    'seconds': 1.,
    'minutes': 60.,
    'hours': 60. * 60.,
    'days': 24. * 60. * 60.,
    'years': YEAR_LENGTH * 24. * 60. * 60.
}
SECONDS_TO_X = {key: 1. / val for key, val in X_TO_SECONDS.items()}


def convert_time(time_value, in_unit, out_unit):
    return time_value * X_TO_SECONDS[in_unit] * SECONDS_TO_X[out_unit]


def format_time(time_value, in_unit='seconds'):
    all_units = X_TO_SECONDS.keys()
    val_in_all_units = {u: convert_time(time_value, in_unit, u) for u in all_units}
    valid_units = {u: v for u, v in val_in_all_units.items() if v >= 1.}
    if valid_units:
        best_unit = min(valid_units, key=valid_units.get)
    else:
        best_unit = 'seconds'
    return val_in_all_units[best_unit], best_unit
