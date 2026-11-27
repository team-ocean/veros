"""Reproducibility metadata for numerical research reports."""

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(config, *, source_directory=None):
    import veros
    from veros import runtime_settings

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()

    package_root = Path(veros.__file__).resolve().parent
    contents = {
        "veros/" + path.relative_to(package_root).as_posix(): path.read_bytes() for path in package_root.rglob("*.py")
    }
    for name in git("ls-files", "--cached", "--others", "--exclude-standard", "--", "benchmarks/ad_demos").splitlines():
        if name.endswith(".py") and name.startswith("benchmarks/ad_demos/"):
            contents[name] = (ROOT / name).read_bytes()
    sources = {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}
    snapshot = None
    if source_directory is not None:
        fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        directory = Path(source_directory)
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = directory / f"{fingerprint}.zip"
        if not snapshot.exists():
            temporary = snapshot.with_suffix(".tmp")
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, content in sorted(contents.items()):
                    archive.writestr(name, content)
            temporary.replace(snapshot)
    versions = {}
    for package in (
        "veros",
        "jax",
        "jaxlib",
        "numpy",
        "scipy",
        "h5netcdf",
        "h5py",
        "mpi4py",
        "mpi4jax",
        "psutil",
        "cython",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    assets = {}
    if config["setup"] == "global_4deg":
        from veros.tools.assets import ASSET_DIRECTORY

        manifest = json.loads((package_root / "setups/global_4deg/assets.json").read_text())
        for key, entry in manifest.items():
            path = Path(ASSET_DIRECTORY) / "global_4deg" / entry["url"].rsplit("/", 1)[-1]
            assets[key] = dict(entry, path=str(path), present=path.exists())
            if path.exists():
                assets[key].update(bytes=path.stat().st_size, sha256=file_hash(path), actual_md5=file_hash(path, "md5"))
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            None,
        )
    return dict(
        commit=git("rev-parse", "HEAD"),
        working_tree=git("status", "--short"),
        source_sha256=sources,
        source_snapshot=str(snapshot) if snapshot is not None else None,
        versions=versions,
        runtime={
            name: getattr(runtime_settings, name)
            for name in (
                "backend",
                "device",
                "float_type",
                "use_special_tdma",
                "linear_solver",
            )
        },
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        cpu_model=cpu_model,
        logical_cpus=os.cpu_count(),
        devices=[str(device) for device in sys.modules["jax"].devices()] if "jax" in sys.modules else None,
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        environment={
            key: os.environ.get(key)
            for key in (
                "VEROS_BACKEND",
                "VEROS_DEVICE",
                "VEROS_USE_SPECIAL_TDMA",
                "VEROS_ASSET_DIR",
                "JAX_PLATFORMS",
                "JAX_COMPILATION_CACHE_DIR",
                "JAX_ENABLE_COMPILATION_CACHE",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
            )
        },
        assets=assets,
        seeds=dict(
            observation_posterior=27,
            observation_forward=71,
            statistical_validation=list(range(8)),
            baseline_diagnosis=42,
        ),
        native_extensions={
            "veros/" + path.relative_to(package_root).as_posix(): file_hash(path)
            for path in sorted((package_root / "core/special").glob("*.so"))
        },
    )
