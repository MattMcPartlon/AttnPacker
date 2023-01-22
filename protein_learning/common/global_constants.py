"""Global constants"""
import logging
import sys
from datetime import datetime
from time import localtime, strftime

import torch

TRAINING, TESTING, VALIDATION = "training", 'testing', 'validation'
STATS, MODELS, PARAMS, LOGS, CHECKPOINTS = 'stats', 'models', 'params', 'logs', 'checkpoints'
NAME, PATH, EXT = 'name', 'path', 'ext'
REZERO_INIT = 0.01
MAX_SEQ_LEN = 800
START_TIME = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")

# Creating and Configuring Logger
logging.basicConfig(
    stream=sys.stdout,
    level=logging.WARNING
)
get_logger = lambda name: logging.getLogger(name)
get_current_time = lambda: strftime("%Y-%m-%d %H:%M:%S", localtime())

JIT_OPTIONS_SET = False


def _set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    global JIT_OPTIONS_SET
    if JIT_OPTIONS_SET == False:
        # flags required to enable jit fusion kernels
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        # if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
        #     # nvfuser
        #     torch._C._jit_set_profiling_executor(True)
        #     torch._C._jit_set_profiling_mode(True)
        #     torch._C._jit_override_can_fuse_on_cpu(False)
        #     torch._C._jit_override_can_fuse_on_gpu(False)
        #     torch._C._jit_set_texpr_fuser_enabled(False)
        #     torch._C._jit_set_nvfuser_enabled(True)
        #     torch._C._debug_set_autodiff_subgraph_inlining(False)
        # else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)  # noqa
        torch._C._jit_set_profiling_executor(False)  # noqa
        torch._C._jit_override_can_fuse_on_cpu(True)  # noqa
        torch._C._jit_override_can_fuse_on_gpu(True)  # noqa

        JIT_OPTIONS_SET = True


