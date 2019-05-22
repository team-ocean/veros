import os
import shutil

import hashlib

import requests
import ruamel.yaml as yaml
from loguru import logger

try:
    import urlparse
except ImportError:
    import urllib.parse as urlparse

ASSET_DIRECTORY = os.environ.get('VEROS_ASSET_DIR') or os.path.join(os.path.expanduser('~'), '.veros', 'assets')


class AssetError(Exception):
    pass


def get_assets(asset_id, asset_file):
    """Handles automatic download and verification of external assets (such as forcing files).

    By default, assets are stored in ``$HOME/.veros/assets`` (can be overwritten by setting
    ``VEROS_ASSET_DIR`` environment variable to the desired location).

    Arguments:

       asset_id (str): Identifier of the collection of assets. Should be unique for each setup.
       asset_file (str): YAML file containing URLs and (optionally) MD5 hashsums of each asset.

    Returns:

        A ``dict`` mapping identifier of each asset to file name on disk.

    Example:

       >>> get_assets('mysetup', 'assets.yml')
       {
           "forcing": "/home/user/.veros/assets/mysetup/mysetup_forcing.h5",
           "initial_conditions": "/home/user/.veros/assets/mysetup/initial.h5"
       }

    In this case, ``assets.yml`` contains::

       forcing:
           url: https://mywebsite.com/veros_assets/mysetup_forcing.h5
           md5: ef3be0a58782771c8ee5a6d0206b87f6

       initial_conditions:
           url: https://mywebsite.com/veros_assets/initial.h5
           md5: d1b4e0e199d7a5883cf7c88d3d6bcb28

    """
    with open(asset_file, 'r') as f:
        assets = yaml.safe_load(f)

    asset_dir = os.path.join(ASSET_DIRECTORY, asset_id)

    if not os.path.isdir(asset_dir):
        try: # possible race-condition
            os.makedirs(asset_dir)
        except OSError:
            if os.path.isdir(asset_dir):
                pass

    def get_asset(url, md5=None):
        target_filename = os.path.basename(urlparse.urlparse(url).path)
        target_path = os.path.join(asset_dir, target_filename)

        if not os.path.isfile(target_path) or (md5 is not None and _filehash(target_path) != md5):
            logger.info('Downloading asset {} ...', target_filename)
            _download_file(url, target_path)

        if md5 is not None and _filehash(target_path) != md5:
            raise AssetError('Mismatching MD5 checksum on asset %s' % target_filename)

        return target_path

    return {key: get_asset(val['url'], val.get('md5', None)) for key, val in assets.items()}


def _download_file(url, target_path, timeout=10):
    """Download a file and save it to a folder"""
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with open(target_path, 'wb') as dst:
            shutil.copyfileobj(response.raw, dst)

    return target_path


def _filehash(path):
    hash_md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
