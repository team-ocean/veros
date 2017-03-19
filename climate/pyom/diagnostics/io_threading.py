import threading
import contextlib
import logging
from netCDF4 import Dataset

from climate.pyom import pyom_method

_io_locks = {}
def _add_to_locks(file_id):
    """
    If there is no lock for file_id, create one
    """
    if not file_id in _io_locks:
        _io_locks[file_id] = threading.Event()
        _io_locks[file_id].set()

def _wait_for_disk(pyom, file_id):
    """
    Wait for the lock of file_id to be released
    """
    logging.debug("Waiting for lock {} to be released".format(file_id))
    _add_to_locks(file_id)
    lock_released = _io_locks[file_id].wait(pyom.io_timeout)
    if not lock_released:
        raise RuntimeError("Timeout while waiting for disk IO to finish")

def _write_to_disk(ncfile, file_id):
    """
    Sync netCDF data to disk, close file handle, and release lock
    """
    ncfile.sync()
    ncfile.close()
    if not file_id is None:
        _io_locks[file_id].set()

@pyom_method
@contextlib.contextmanager
def threaded_netcdf(pyom, filepath, mode):
    """
    If using IO threads, start a new thread to write the netCDF data to disk.
    """
    if pyom.use_io_threads:
        _wait_for_disk(pyom, filepath)
        _io_locks[filepath].clear()
    nc_dataset = Dataset(filepath, mode)
    try:
        yield nc_dataset
    finally:
        if pyom.use_io_threads:
            io_thread = threading.Thread(target=_write_to_disk, args=(nc_dataset, filepath))
            io_thread.start()
        else:
            _write_to_disk(nc_dataset, filepath)
