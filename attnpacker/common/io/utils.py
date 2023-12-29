from typing import List, Any

import numpy as np


def load_npy(path) -> Any:
    """Load .npy data type"""
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e1:
        try:
            data = np.load(path, allow_pickle=True, encoding='latin1')
        except Exception as e2:
            print(f'could not load file {path}\n'
                  f' exception when loading with default encoding {e1}'
                  f'\n exception when loading with latin1 {e2}')
            raise e1
    try:
        data = data.item()
    except:  # noqa
        pass
    return data


def parse_arg_file(path) -> List[str]:
    """Parse a param file"""
    args = []
    with open(path, 'r') as f:
        for x in f:  # noqa
            line = x.strip()
            if len(line.strip()) > 0 and not line.startswith("#"):
                arg = line.split(' ')
                for a in arg:
                    args.append(a)
    return args
