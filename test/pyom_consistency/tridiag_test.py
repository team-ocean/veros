import pytest
import numpy as np

from veros import runtime_settings
from veros.pyom_compat import load_pyom


@pytest.mark.skipif(runtime_settings.backend != "jax", reason="Must use JAX backend")
@pytest.mark.parametrize("use_ext", [True, False])
def test_solve_tridiag_jax(pyom2_lib, use_ext):
    from veros.core.operators import solve_tridiagonal_jax
    from veros.core.utilities import create_water_masks

    pyom_obj = load_pyom(pyom2_lib)

    nx, ny, nz = 70, 60, 50
    a, b, c, d = (np.random.randn(nx, ny, nz) for _ in range(4))
    kbot = np.random.randint(0, nz, size=(nx, ny))

    out_pyom = np.zeros((nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            ks = kbot[i, j] - 1
            ke = nz

            if ks < 0:
                continue

            out_pyom[i, j, ks:ke] = pyom_obj.solve_tridiag(
                a=a[i, j, ks:ke], b=b[i, j, ks:ke], c=c[i, j, ks:ke], d=d[i, j, ks:ke], n=ke - ks
            )

    _, water_mask, edge_mask = create_water_masks(kbot, nz)
    out_vs = solve_tridiagonal_jax(a, b, c, d, water_mask, edge_mask, use_ext=use_ext)

    np.testing.assert_allclose(out_pyom, out_vs)


@pytest.mark.skipif(runtime_settings.backend != "numpy", reason="Must use NumPy backend")
def test_solve_tridiag_numpy(pyom2_lib):
    from veros.core.operators import solve_tridiagonal_numpy
    from veros.core.utilities import create_water_masks

    pyom_obj = load_pyom(pyom2_lib)

    nx, ny, nz = 70, 60, 50
    a, b, c, d = (np.random.randn(nx, ny, nz) for _ in range(4))
    kbot = np.random.randint(0, nz, size=(nx, ny))

    out_pyom = np.zeros((nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            ks = kbot[i, j] - 1
            ke = nz

            if ks < 0:
                continue

            out_pyom[i, j, ks:ke] = pyom_obj.solve_tridiag(
                a=a[i, j, ks:ke], b=b[i, j, ks:ke], c=c[i, j, ks:ke], d=d[i, j, ks:ke], n=ke - ks
            )

    _, water_mask, edge_mask = create_water_masks(kbot, nz)
    out_vs = solve_tridiagonal_numpy(a, b, c, d, water_mask, edge_mask)

    np.testing.assert_allclose(out_pyom, out_vs)
