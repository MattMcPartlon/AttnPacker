import _datetime as datetime
import time

import torch
import logging
import sys
from typing import NamedTuple
import torch
from torch import sin, cos, atan2, acos
from functools import wraps

VERBOSE = True


def update_named_tuple(cls, instance: NamedTuple, **kwargs):
    instance_kwargs = instance._asdict()  # noqa
    instance_kwargs.update(kwargs)
    return cls(**instance_kwargs)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def uniq(arr):
    return list({el: True for el in arr}.keys())


def to_order(degree):
    return 2 * degree + 1


def map_values(fn, d):
    return {k: fn(v) for k, v in d.items()}


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


def print_time():
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


def calc_tm_torch(deviations, norm_len=None, reduce=True):
    norm_len = norm_len if norm_len else deviations.numel()
    if norm_len <= 15:
        d0 = 0.5
    else:
        d0 = 1.24 * ((norm_len - 15.0) ** (1. / 3.)) - 1.8
    per_res_scores = 1 / (1 + (torch.square(deviations) / (d0 ** 2)))
    return torch.mean(per_res_scores) if reduce else per_res_scores


def RMSD(c1, c2):
    return torch.sqrt(torch.nn.functional.mse_loss(c1.squeeze(), c2.squeeze())).item()


def TM(c1, c2, norm_len=None):
    return calc_tm_torch(torch.norm(c1.squeeze() - c2.squeeze(), dim=-1), norm_len=norm_len)


def timed(message="", self=None):
    def time_decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter_ns()
            if self:
                value = self.func(*args, **kwargs)
            else:
                value = func(*args, **kwargs)

            stop = time.perf_counter_ns()
            if VERBOSE:
                print(message, "in", (stop - start) / 10 ** 9, "seconds.")
            return value

        return wrapper

    return time_decorator


def log(logger: logging.Logger, message, *args, verbose=logging.DEBUG, prefix='', std_out=False):
    try:
        msg = prefix + message.format(*args)
    except:  # noqa
        msg = prefix + message
    if std_out:
        print(msg)
    logger.log(verbose, msg)


def configure_logger(log_dir, level=logging.DEBUG):
    if log_dir is not None:
        logging.basicConfig(filename=log_dir, level=level,
                            format='%(asctime)s %(levelname)s %(name)s %(message)s', filemode='w+')
    else:
        logging.basicConfig(stream=sys.stdout, level=level,
                            format='%(asctime)s %(levelname)s %(name)s %(message)s', filemode='w+')


def cast_torch_tensor(fn):
    @wraps(fn)
    def inner(t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.get_default_dtype())
        return fn(t)

    return inner


@cast_torch_tensor
def rot_z(gamma):
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)


@cast_torch_tensor
def rot_y(beta):
    return torch.tensor([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype=beta.dtype)


def rot(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)
