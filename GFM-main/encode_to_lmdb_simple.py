import lmdb
import os
from safetensors.numpy import save

def save_bands_to_safetensor(bands_dict):
    """
    Writes all bands in a dictionary to safetensor format.

    :param bands_dict: Dictionary {Bandname: NumPy-Array}
    :return: Bytes object as safetensor
    """
    return save(bands_dict)


def write_to_lmdb(db, key, safetensor_data):
    """
    Writes a multidimensional safetensor into an lmdb. Map-size is adapted automatically

    :param db: LMDB
    :param key: Key for the safetensor
    :param bands_dict: Dictionary with {Bandname: NumPy-Array}
    """
    success = False

    while not success:
        txn = db.begin(write=True)
        try:
            txn.put(key.encode(), safetensor_data)
            txn.commit()
            success = True
            print(f"TIFF '{key}' written successfully to LMDB!")
        except lmdb.MapFullError:
            txn.abort()
            curr_limit = db.info()['map_size']
            new_limit = curr_limit * 2
            print(f"Full storage! Raise current lmdb-size by factor 2 {new_limit >> 20}MB ...")
            db.set_mapsize(new_limit)

def create_or_open_lmdb(lmdb_path, size=None):
    """
    Creates a new lmdb database or opens an existing one

    :param lmdb_path: path to the lmdb
    :param size: maximal storage size in bytes (optional)
    :return: lmdb environment
    """
    if os.path.exists(lmdb_path):
        print(f"Open existing LMDB: {lmdb_path}")

        temp_env = lmdb.open(lmdb_path, readonly=True)
        existing_size = temp_env.info()['map_size']
        temp_env.close()

        map_size = size if size else existing_size
        print(f"Use existing map_size: {map_size >> 20}MB")

        return lmdb.open(lmdb_path, map_size=map_size)
    else:
        default_size = 10 * 1024 * 1024
        map_size = size if size else default_size
        print(f"Create new LMDB: {lmdb_path} with {map_size >> 20}MB storage space")

        return lmdb.open(lmdb_path, map_size=map_size)