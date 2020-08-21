import scipy.ndimage

from veros.core import utilities
from veros.core.operators import numpy as np

# fall back to vanilla NumPy for some operations
import numpy as onp


def isleperim(kmt, enable_cyclic_x, verbose=False):
    kmt = utilities.enforce_boundaries(kmt, enable_cyclic_x)
    kmt = onp.asarray(kmt)

    structure = onp.ones((3, 3))  # merge diagonally connected land masses

    # find all land masses
    labelled, _ = scipy.ndimage.label(kmt == 0, structure=structure)

    # find and set perimeter
    land_masses = labelled > 0
    inner = scipy.ndimage.binary_dilation(land_masses, structure=structure)

    perimeter = onp.logical_xor(inner, land_masses)
    labelled[perimeter] = -1

    # match wrapping periodic land masses
    if enable_cyclic_x:
        west_slice = onp.array(labelled[2])
        east_slice = onp.array(labelled[-2])

        for west_label in onp.unique(west_slice[west_slice > 0]):
            east_labels = onp.unique(east_slice[west_slice == west_label])
            east_labels = east_labels[~onp.isin(east_labels, [west_label, -1])]
            if not east_labels.size:
                # already labelled correctly
                continue
            assert len(onp.unique(east_labels)) == 1, (west_label, east_labels)
            labelled[labelled == east_labels[0]] = west_label

    labelled = utilities.enforce_boundaries(labelled, enable_cyclic_x)
    labelled = onp.asarray(labelled)

    # label landmasses in a way that is consistent with pyom
    labels = onp.unique(labelled[labelled > 0])

    label_idx = {}
    for label in labels:
        # find index of first island cell, scanning west to east, north to south
        label_idx[label] = onp.argmax(labelled[:, ::-1].T == label)

    sorted_labels = list(sorted(labels, key=lambda i: label_idx[i]))

    # ensure labels are numbered consecutively
    relabelled = onp.array(labelled)
    for new_label, label in enumerate(sorted_labels, 1):
        if label == new_label:
            continue
        relabelled[labelled == label] = new_label

    return np.asarray(relabelled)
