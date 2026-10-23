"""Archived numerical sources must describe the imported physical package."""

import hashlib
from pathlib import Path
import zipfile


def test_provenance_captures_an_imported_package_outside_the_checkout(tmp_path, monkeypatch):
    import veros
    from benchmarks.ad_demos.provenance import collect

    package = tmp_path / "frozen" / "veros"
    native = package / "core/special/solver.so"
    native.parent.mkdir(parents=True)
    contents = {
        "__init__.py": b"version = 'frozen'\n",
        "model.py": b"coefficient = 17\n",
        "core/special/solver.so": b"frozen native extension",
    }
    for name, raw in contents.items():
        (package / name).write_bytes(raw)
    monkeypatch.setattr(veros, "__file__", str(package / "__init__.py"))
    result = collect(dict(setup="fixture"), source_directory=tmp_path / "sources")
    expected = {"veros/" + name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()}
    assert {name: value for name, value in result["source_sha256"].items() if name.startswith("veros/")} == {
        name: value for name, value in expected.items() if name.endswith(".py")
    }
    assert result["native_extensions"] == {"veros/core/special/solver.so": expected["veros/core/special/solver.so"]}
    with zipfile.ZipFile(result["source_snapshot"]) as archive:
        assert archive.read("veros/model.py") == contents["model.py"]
        assert "veros/core/friction.py" not in archive.namelist()
        assert (
            archive.read("benchmarks/ad_demos/provenance.py")
            == Path(__file__).resolve().parents[1].joinpath("benchmarks/ad_demos/provenance.py").read_bytes()
        )


def test_checkout_provenance_retains_numerical_source_names():
    import veros
    from benchmarks.ad_demos.provenance import collect

    result = collect(dict(setup="fixture"))
    package = Path(veros.__file__).resolve().parent
    assert (
        result["source_sha256"]["veros/core/friction.py"]
        == hashlib.sha256((package / "core/friction.py").read_bytes()).hexdigest()
    )
