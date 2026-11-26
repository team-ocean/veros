import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


ROOT = Path(__file__).resolve().parent
NATIVE_PACKAGE = Path("veros/core/special")
MINIMUM_VERSIONS = {"numpy": "1.13", "requests": "2.18", "jax": "0.5.0"}


def read_requirements(path):
    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        requirement = Requirement(line)
        requirement.specifier = SpecifierSet(str(requirement.specifier).replace("==", "<="))
        if requirement.name in MINIMUM_VERSIONS:
            requirement.specifier &= SpecifierSet(f">={MINIMUM_VERSIONS[requirement.name]}")
        requirements.append(requirement)
    return requirements


class VerosMetadataHook(MetadataHookInterface):
    def update(self, metadata):
        metadata["dependencies"] = list(map(str, read_requirements(ROOT / "requirements.txt")))
        jax_requirements = read_requirements(ROOT / "requirements_jax.txt")
        jax = next(requirement for requirement in jax_requirements if requirement.name == "jax")
        metadata["optional-dependencies"] = {
            # pytest 9.2 breaks pytest-forked (https://github.com/pytest-dev/pytest-forked/issues/113)
            "test": ["pytest<9.2", "pytest-cov", "pytest-forked", "xarray"],
            "jax": [*map(str, jax_requirements), str(jax).replace("jax", "jaxlib", 1)],
        }


def env_enabled(name):
    return os.environ.get(name, "").lower() in ("1", "true", "on")


class NativeBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        self.build_dir = TemporaryDirectory(prefix="veros-native-")
        build = Path(self.build_dir.name)
        extensions = self.build_extensions(build)
        if not extensions:
            return

        if version == "editable":
            for extension in extensions:
                shutil.copy2(extension, ROOT / NATIVE_PACKAGE / extension.name)
        else:
            for extension in extensions:
                build_data["force_include"][str(extension)] = (NATIVE_PACKAGE / extension.name).as_posix()
            build_data["infer_tag"] = True
            build_data["pure_python"] = False

    def finalize(self, version, build_data, artifact_path):
        self.build_dir.cleanup()

    def build_extensions(self, build):
        import jaxlib
        import nanobind

        require_cpu = env_enabled("VEROS_REQUIRE_CPU_EXT") or env_enabled("VEROS_REQUIRE_CYTHON_EXT")
        require_cuda = env_enabled("VEROS_REQUIRE_CUDA_EXT")
        output = build / "extensions"
        configure = [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={output}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dnanobind_DIR={nanobind.cmake_dir()}",
            f"-DJAXLIB_INCLUDE_DIR={Path(jaxlib.__file__).with_name('include')}",
        ]
        if sys.platform != "win32" and "CMAKE_GENERATOR" not in os.environ:
            configure.extend(["-G", "Ninja"])
        cuda_root = os.environ.get("CUDAHOME") or os.environ.get("CUDA_ROOT")
        if cuda_root:
            configure.append(f"-DCUDAToolkit_ROOT={cuda_root}")

        try:
            subprocess.run(configure, check=True)
        except subprocess.CalledProcessError:
            if require_cpu or require_cuda:
                raise
            self.app.display_warning("Could not configure the optional native TDMA extensions")
            return []

        targets = {"tdma_cpu_": require_cpu}
        if (build / "cuda-enabled").exists():
            targets["tdma_cuda_"] = require_cuda
        elif require_cuda:
            raise RuntimeError("The CUDA TDMA extension requires a working CUDA toolkit")

        extensions = []
        for target, required in targets.items():
            extension = output / f"{target}{sysconfig.get_config_var('EXT_SUFFIX')}"
            try:
                subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--target", target], check=True)
                if not extension.is_file():
                    raise RuntimeError(f"The build did not produce {extension.name}")
            except (RuntimeError, subprocess.CalledProcessError):
                if required:
                    raise
                self.app.display_warning(f"Could not build the optional {target} extension")
            else:
                extensions.append(extension)
        return extensions
