import os
import shutil
import logging
import hashlib

import requests
import ruamel.yaml as yaml

try:
    import urlparse
except ImportError:
    import urllib.parse as urlparse

ASSET_DIRECTORY = os.environ.get("VEROS_ASSET_DIR") or os.path.join(os.path.expanduser("~"), ".veros", "assets")


class AssetError(Exception):
    pass


def get_assets(basedir, asset_file):
    with open(asset_file, "r") as f:
        assets = yaml.safe_load(f)

    asset_dir = os.path.join(ASSET_DIRECTORY, basedir)

    if not os.path.isdir(asset_dir):
        os.makedirs(asset_dir)

    def get_asset(url, md5=None):
        target_filename = os.path.basename(urlparse.urlparse(url).path)
        target_path = os.path.join(asset_dir, target_filename)

        if not os.path.isfile(target_path) or (md5 is not None and _filehash(target_path) != md5):
            logging.info("Downloading asset %s ...", target_filename)
            _download_file(url, target_path)

        if md5 is not None and _filehash(target_path) != md5:
            raise AssetError("Mismatching MD5 checksum on asset %s" % target_filename)

        return target_path

    return {key: get_asset(val["url"], val.get("md5", None)) for key, val in assets.items()}


def _download_file(url, target_path, timeout=10):
    """Download a file and save it to a folder"""
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with open(target_path, "wb") as dst:
            shutil.copyfileobj(response.raw, dst)

    return target_path


def _filehash(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
