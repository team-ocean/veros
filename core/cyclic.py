def setcyclic_x(array):
    """
    set cyclic boundary conditions on the first axis
    """
    array[-2:, ...] = array[2:4, ...]
    array[:2, ...] = array[-4:-2, ...]


def setcyclic_p(array):
    """
    set cyclic boundary conditions on the third axis
    """
    array[:, :, 1] = array[:, :, -2]
    array[:, :, -1] = array[:, :, 2]


def setcyclic_xp(array):
    """
    set cyclic x- and p-boundary conditions for array
    """
    setcyclic_p(array)
    setcyclic_x(array)
