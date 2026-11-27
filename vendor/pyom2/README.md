# PyOM consistency reference

The archive contains [PyOM 2.3.1](https://github.com/ceden/pyOM2/releases/tag/2.3.1) at commit `4518280bd7a179e1d7ce84c2d84ec2614fe12789`. It was generated with `git archive 2.3.1 | gzip -n`.

The patches apply upstream’s [Python build repair](https://github.com/ceden/pyOM2/commit/37407dd64770ef1231104aadf0595920a5274b5a), preserve double-precision model arithmetic and the unequal-timestep free-surface correction, and repair uninitialized intermediates and the upper W-grid DST3 stencil. The tests exercise the shared hydrostatic configurations.

Run the build from the Veros repository root with gfortran, BLAS, LAPACK and the Veros test environment available.

```sh
python -m pip install meson ninja
pyom_build=$(mktemp -d)
tar -xzf vendor/pyom2/pyOM2.3.1.tar.gz -C "$pyom_build"
cp vendor/pyom2/pyOM2_site_specific_meson "$pyom_build/site_specific.mk_"
for patch in "$PWD"/vendor/pyom2/patches/*.patch; do
    git -C "$pyom_build" apply "$patch"
done
make -C "$pyom_build/py_src" HOSTTYPE= pyOM_code.so
pytest test/pyom_consistency --pyom2-lib "$pyom_build"/py_src/pyOM_code.*.so
```
