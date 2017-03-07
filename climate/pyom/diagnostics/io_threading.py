import threading
import contextlib

_io_locks = {}
def _add_to_locks(file_id, lock_immediately=False):
    """
    If there is no lock for file_id, create one
    """
    if not file_id in _io_locks:
        _io_locks[file_id] = threading.Event()
        if not lock_immediately:
            _io_locks[file_id].set()

def _wait_for_disk(file_id, pyom):
    """
    Wait for the lock of file_id to be released
    """
    _add_to_locks(file_id)
    lock_released = _io_locks[file_id].wait(pyom.io_timeout)
    if not lock_released:
        raise RuntimeError("Timeout while waiting for disk IO to finish")

def _write_to_disk(ncfile, file_id=None):
    """
    Sync netCDF data to disk, close file handle. and release lock
    """
    ncfile.sync()
    ncfile.close()
    if not file_id is None:
        _io_locks[file_id].set()


@contextlib.contextmanager
def threaded_netcdf(ncfile, pyom, file_id=None):
    """
    If using IO threads, start a new thread to write the netCDF data to disk.
    Note that locking only occurs when file_id is given.
    """
    if not file_id is None:
        _wait_for_disk(file_id, pyom)
        _io_locks[file_id].clear()
    try:
        yield ncfile
    finally:
        if pyom.use_io_threads:
            if not file_id is None:
                _add_to_locks(file_id, lock_immediately=True)
                _io_locks[file_id].clear()
            io_thread = threading.Thread(target=_write_to_disk, args=(ncfile, file_id))
            io_thread.start()
        else:
            _write_to_disk(ncfile, file_id)
