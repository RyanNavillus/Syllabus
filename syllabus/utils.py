from itertools import product
from typing import Union

import numpy as np


def decorate_all_functions(function_decorator):
    def decorator(cls):
        for base_cls in cls.__bases__:
            for name, obj in vars(base_cls).items():
                parent_func = getattr(base_cls, name)
                child_func = getattr(cls, name)

                # Only apply decorator to functions not overridden by subclass.
                if callable(obj) and child_func == parent_func:
                    setattr(cls, name, function_decorator(obj))
            return cls
    return decorator


class UsageError(Exception):
    pass


def enumerate_axes(list_or_size: Union[np.ndarray, int]):
    if isinstance(list_or_size, int) or isinstance(list_or_size, np.int64):
        return tuple(range(list_or_size))
    elif isinstance(list_or_size, list) or isinstance(list_or_size, np.ndarray):
        return tuple(product(*[enumerate_axes(x) for x in list_or_size]))
    else:
        raise NotImplementedError(f"{type(list_or_size)}")
