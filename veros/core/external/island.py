try: # python 2.x
    import Queue as queue
except ImportError: # python 3.x
    import queue
import logging

from .. import cyclic, utilities
from ... import veros_method

OFFMAP = -1
LAND = 1
OCEAN = 0


@veros_method
def isleperim(vs, kmt, verbose=False):
    """
    Island and Island Perimeter boundary mapping routines
    """

    if verbose:
        logging.info(" Finding perimeters of all land masses")
        if vs.enable_cyclic_x:
            logging.info(" using cyclic boundary conditions")

    """
    copy kmt to map changing notation
    initially, 0 means ocean and 1 means unassigned land in map
    as land masses are found, they are labeled 2, 3, 4, ...,
    and their perimeter ocean cells -2, -3, -4, ...,
    when no land points remain unassigned, land mass numbers are
    reduced by 1 and their perimeter ocean points relabelled accordingly
    """
    boundary_map = utilities.where(vs, kmt > 0, OCEAN, LAND)
    if vs.backend_name == "bohrium":
        vs.flush()
        boundary_map = boundary_map.copy2numpy()

    """
    find unassigned land points and expand them to continents
    """
    imt, jmt = vs.nx + 4, vs.ny + 4
    island_queue = queue.Queue()
    label = 2
    nippts = [0]
    jnorth = jmt - 1
    if vs.enable_cyclic_x:
        iwest = 1
        ieast = imt - 2
    else:
        iwest = 0
        ieast = imt - 1

    for j in range(jnorth, -1, -1):
        for i in range(iwest, ieast):
            if boundary_map[i, j] == LAND:
                island_queue.put((i, j))
                expand(vs, boundary_map, label, island_queue, nippts)
                if verbose:
                    logging.debug(" found island {} with {} perimeter points"
                                  .format(label-1, nippts[-1]))
                label += 1
                nippts.append(0)
    nisle = label - 2

    boundary_map[iwest:ieast + 1, :jnorth + 1] += -np.sign(boundary_map[iwest:ieast + 1, :jnorth + 1])
    if vs.enable_cyclic_x:
        boundary_map[0, :] = boundary_map[imt - 2, :]
        boundary_map[imt - 1, :] = boundary_map[1, :]

    if verbose:
        logging.info(" Island perimeter statistics:")
        logging.info("  number of land masses is {}".format(nisle))
        logging.info("  number of total island perimeter points is {}".format(sum(nippts)))
    return np.asarray(boundary_map)


@veros_method
def expand(vs, boundary_map, label, island_queue, nippts):
    """
    This function uses a "flood fill" algorithm
    to expand one previously unmarked land
    point to its entire connected land mass and its perimeter
    ocean points. Diagonally adjacent land points are
    considered connected. Perimeter "collisions" (i.e.,
    ocean points that are adjacent to two unconnected
    land masses) are detected and error messages generated.
    """
    imt, jmt = vs.nx + 4, vs.ny + 4

    # main loop: pop a candidate point off the queue and process it
    while not island_queue.empty():
        (i, j) = island_queue.get()
        # case: (i,j) is off the map
        if i == OFFMAP or j == OFFMAP:
            continue
        # case: (i,j) is already labeled for this land mass
        elif boundary_map[i, j] == label:
            continue
        # case: (i,j) is an ocean perimeter point of this land mass
        elif boundary_map[i, j] == -label:
            continue
        # case: (i,j) is an unassigned land point
        elif boundary_map[i, j] == LAND:
            boundary_map[i, j] = label
            island_queue.put((i, jn_isl(j, jmt)))
            island_queue.put((ie_isl(vs, i, imt), jn_isl(j, jmt)))
            island_queue.put((ie_isl(vs, i, imt), j))
            island_queue.put((ie_isl(vs, i, imt), js_isl(j)))
            island_queue.put((i, js_isl(j)))
            island_queue.put((iw_isl(vs, i, imt), js_isl(j)))
            island_queue.put((iw_isl(vs, i, imt), j))
            island_queue.put((iw_isl(vs, i, imt), jn_isl(j, jmt)))
            continue
        # case: (i,j) is a perimeter ocean point of another mass
        elif boundary_map[i, j] < 0:
            continue
        # case: (i,j) is an unflagged ocean point adjacent to this land mass
        elif boundary_map[i, j] == OCEAN:
            boundary_map[i, j] = -label
            nippts[-1] += 1
        # case: (i,j) is probably labeled for another land mass
        # ************ this case should not happen ************
        else:
            raise RuntimeError("point ({},{}) is labeled for both land masses {} and {}"
                               .format(i, j, boundary_map[i, j] - 1, label - 1))


def jn_isl(j, jmt):
    return j + 1 if j < jmt - 1 else OFFMAP


def js_isl(j):
    return j - 1 if j > 0 else OFFMAP


def ie_isl(vs, i, imt):
    if vs.enable_cyclic_x:
        return i + 1 if i < imt - 2 else i + 1 - imt + 2
    else:
        return i + 1 if i < imt - 1 else OFFMAP


def iw_isl(vs, i, imt):
    if vs.enable_cyclic_x:
        return i - 1 if i > 1 else i - 1 + imt - 2
    else:
        return i - 1 if i > 0 else OFFMAP
