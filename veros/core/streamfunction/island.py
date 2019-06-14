import numpy
import scipy.ndimage

from ... import veros_method, runtime_settings as rs
from .. import utilities


@veros_method
def isleperim(vs, kmt, verbose=False):
    utilities.enforce_boundaries(vs, kmt)

    if rs.backend == 'bohrium':
        kmt = kmt.copy2numpy()

    structure = numpy.ones((3, 3))  # merge diagonally connected land masses

    # find all land masses
    labelled, _ = scipy.ndimage.label(kmt == 0, structure=structure)

    # find and set perimeter
    land_masses = labelled > 0
    inner = scipy.ndimage.binary_dilation(land_masses, structure=structure)
    perimeter = numpy.logical_xor(inner, land_masses)
    labelled[perimeter] = -1

    # match wrapping periodic land masses
    if vs.enable_cyclic_x:
        west_slice = labelled[2]
        east_slice = labelled[-2]

        for west_label in numpy.unique(west_slice[west_slice > 0]):
            east_labels = numpy.unique(east_slice[west_slice == west_label])
            east_labels = east_labels[~numpy.isin(east_labels, [west_label, -1])]
            if not east_labels.size:
                # already labelled correctly
                continue
            assert len(numpy.unique(east_labels)) == 1, (west_label, east_labels)
            labelled[labelled == east_labels[0]] = west_label

    utilities.enforce_boundaries(vs, labelled)

    # label landmasses in a way that is consistent with pyom
    labels = numpy.unique(labelled[labelled > 0])

    label_idx = {}
    for label in labels:
        # find index of first island cell, scanning west to east, north to south
        label_idx[label] = np.argmax(labelled[:, ::-1].T == label)

    sorted_labels = list(sorted(labels, key=lambda i: label_idx[i]))

    # ensure labels are numbered consecutively
    relabelled = labelled.copy()
    for new_label, label in enumerate(sorted_labels, 1):
        if label == new_label:
            continue
        relabelled[labelled == label] = new_label

    return np.asarray(relabelled)
