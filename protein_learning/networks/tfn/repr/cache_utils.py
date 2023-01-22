import gzip
import os
import pickle
from functools import wraps, lru_cache

from filelock import FileLock

from protein_learning.networks.common.utils import exists


# caching functions
def cache(cache, key_fn):  # noqa
    def cache_inner(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            key_name = key_fn(*args, **kwargs)
            if key_name in cache:
                return cache[key_name]
            res = fn(*args, **kwargs)
            cache[key_name] = res
            return res

        return inner

    return cache_inner


# cache in directory

def cache_dir(dirname, maxsize=128):
    """
    Cache a function with a directory

    :param dirname: the directory path
    :param maxsize: maximum size of the RAM cache (there is no limit for the directory cache)
    """

    def decorator(func):

        @lru_cache(maxsize=maxsize)
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not exists(dirname):
                return func(*args, **kwargs)

            os.makedirs(dirname, exist_ok=True)

            indexfile = os.path.join(dirname, "index.pkl")
            lock = FileLock(os.path.join(dirname, "mutex"))

            with lock:
                index = {}
                if os.path.exists(indexfile):
                    with open(indexfile, "rb") as file:
                        index = pickle.load(file)

                key = (args, frozenset(kwargs), func.__defaults__)

                if key in index:
                    filename = index[key]
                else:
                    index[key] = filename = f"{len(index)}.pkl.gz"
                    with open(indexfile, "wb") as file:
                        pickle.dump(index, file)

            filepath = os.path.join(dirname, filename)

            if os.path.exists(filepath):
                with lock:
                    with gzip.open(filepath, "rb") as file:
                        result = pickle.load(file)
                return result

            print(f"compute {filename}... ", end="", flush=True)
            result = func(*args, **kwargs)
            print(f"save {filename}... ", end="", flush=True)

            with lock:
                with gzip.open(filepath, "wb") as file:
                    pickle.dump(result, file)

            print("done")

            return result

        return wrapper

    return decorator
