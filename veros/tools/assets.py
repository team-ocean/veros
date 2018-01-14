import os
import shutil
import logging

import requests

try:
    import urlparse
except ImportError:
    import urllib.parse as urlparse

ASSET_DIRECTORY = os.environ.get("VEROS_ASSET_DIR") or os.path.join(os.path.expanduser("~"), ".veros", "assets")


def get_assets(basedir, assets):
    asset_dir = os.path.join(ASSET_DIRECTORY, basedir)

    if not os.path.isidr(asset_dir):
        os.makedirs(asset_dir)

    def get_asset(url):
        target_filename = os.path.basename(urlparse.parse(url).path)
        target_path = os.path.join(asset_dir, target_filename)
        if not os.path.isfile(target_path):
            logging.info("Downloading asset %s ...", target_filename)
            _download_file(url, target_path)
        return target_path

    return {key: get_asset(val) for key, val in assets.items()}


def _download_file(url, target_path):
    """Download a file and save it to a folder"""
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with open(target_path, "wb") as dst:
            shutil.copyfileobj(response.raw, dst)

    return target_path
