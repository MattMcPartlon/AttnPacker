"""Helper functions for loading and saving model configs"""
import os
import sys
from argparse import Namespace
from typing import Dict, Union, Optional, Tuple, Any

import numpy as np

from protein_learning.common.global_config import (
    GlobalConfig,
    load_config,
    load_npy,
    save_config,
    print_config,
    make_config,
)
from protein_learning.common.helpers import exists, default
from protein_learning.common.io.utils import parse_arg_file


def get_arg_list_from_std_in():
    arg_file = None if len(sys.argv) == 1 or sys.argv[1] == "-h" else sys.argv[1]
    return parse_arg_file(arg_file) if exists(arg_file) else ["-h"]


def get_args_n_groups(parser, arg_list=None):  # -> Tuple[Namespace, Dict[str, Namespace]]:
    arg_list = arg_list if exists(arg_list) else get_arg_list_from_std_in()
    args = parser.parse_args(arg_list)
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = Namespace(**group_dict)
    return args, arg_groups


def load_args_for_eval(
    global_config_path: str,
    model_config_path: str,
    model_override: Dict,
    global_override: Dict,
    default_model_args: Namespace,
    default_model_arg_groups: Dict,
):
    global_config = load_config(
        curr_config=None,
        config_path=global_config_path,
        **global_override,
    )
    loaded_args = load_npy(model_config_path)
    curr_arg_dict = vars(default_model_args)
    curr_arg_dict.update(loaded_args)

    loaded_arg_groups = {}
    for group_name in default_model_arg_groups:
        group_dict = vars(default_model_arg_groups[group_name])
        for key in group_dict:
            if key not in curr_arg_dict:
                if key != "help":
                    print(f"[WARNING] : no value for key {key}, arg_group : {group_name}")
                continue
            group_dict[key] = curr_arg_dict[key]
        loaded_arg_groups[group_name] = group_dict

    for k, v in default(model_override, {}).items():
        if k in curr_arg_dict:
            curr_arg_dict[k] = v
        for group in loaded_arg_groups:
            if k in loaded_arg_groups[group]:
                loaded_arg_groups[group][k] = v
    loaded_arg_groups = {k: Namespace(**v) for k, v in loaded_arg_groups.items()}
    return global_config, Namespace(**curr_arg_dict), loaded_arg_groups


def override_config_from_path(path, override):
    load_kwargs = load_npy(path)
    load_kwargs["load_state"] = True
    # assert load_kwargs['config_path'] != path, path
    load_kwargs["config_path"] = path
    load_kwargs.update(override)
    return GlobalConfig(**load_kwargs)


def load_n_save_args(
    args,
    arg_groups,
    defaults: Optional[Namespace] = None,
    suffix="model",
    global_override: Dict = None,
    model_override: Dict = None,
    model_config_path: Optional[str] = None,
    save: bool = True,
    force_load: bool = False,
):
    """Load and save model arguments"""
    # Set up model args
    if exists(model_config_path) or getattr(args, "force_override", False):
        pth = default(model_config_path, getattr(args, "global_config_override_path", None))
        print(f"[INFO] loading and overriding global config!")
        print(f"\tmodel config path (given) : {model_config_path}")
        print(f"\tmodel config path (used) {pth}")
        config = override_config_from_path(path=pth, override=default(global_override, dict()))
    else:
        config = make_config(args.model_config)
    if config.load_state or force_load:
        global_override = default(global_override, dict())
        model_override = default(model_override, dict())
        config, args, arg_groups = load_args(
            config,
            args,
            arg_groups,
            defaults=defaults,
            override=global_override,
            model_override=model_override,
            suffix=suffix,
        )
    if save:
        save_args(config, args, suffix=suffix)
    return config, args, arg_groups


def load_args(
    curr_config: GlobalConfig,
    curr_model_args: Namespace,
    arg_groups: Optional[Dict[str, Namespace]] = None,
    defaults: Optional[Namespace] = None,
    suffix: str = "model_args",
    override: Optional[Dict[str, Any]] = None,
    model_override: Optional[Dict[str, Any]] = None,
) -> Union[Tuple[GlobalConfig, Namespace], Tuple[GlobalConfig, Namespace, Dict[str, Namespace]]]:
    """Load model arguments from paths"""
    config = load_config(curr_config, **default(override, {}))
    path = os.path.join(config.param_directory, config.name + f"_{suffix}.npy")
    loaded_args = load_npy(path)
    curr_arg_dict = vars(curr_model_args)
    if "force_override" in curr_arg_dict:
        loaded_args["force_override"] = curr_arg_dict["force_override"]
    curr_arg_dict = vars(defaults) if exists(defaults) else vars(curr_model_args)
    curr_arg_dict.update(loaded_args)

    if not exists(arg_groups):
        return config, Namespace(**curr_arg_dict)

    loaded_arg_groups = {}
    for group_name in arg_groups:
        group_dict = vars(arg_groups[group_name])
        for key in group_dict:
            if key not in curr_arg_dict:
                print(f"[WARNING] : no value for key {key}, arg_group : {group_name}")
                continue
            group_dict[key] = curr_arg_dict[key]
        loaded_arg_groups[group_name] = group_dict

    for k, v in default(model_override, {}).items():
        if k in curr_arg_dict:
            curr_arg_dict[k] = v
        for group in loaded_arg_groups:
            if k in loaded_arg_groups[group]:
                loaded_arg_groups[group][k] = v
    loaded_arg_groups = {k: Namespace(**v) for k, v in loaded_arg_groups.items()}
    return config, Namespace(**curr_arg_dict), loaded_arg_groups


def save_args(config: GlobalConfig, args: Namespace, suffix: str = "model_args"):
    """Save model arguments/config"""
    path = os.path.join(config.param_directory, config.name + f"_{suffix}.npy")
    np.save(path, vars(args))
    save_config(config)


def print_args(
    config: Optional[GlobalConfig], args: Optional[Namespace], arg_groups: Optional[Dict[str, Namespace]] = None
):
    """Print model args"""
    if exists(config):
        print("---------- GLOBAL CONFIG ----------")
        print_config(config)

    print("---------- MODEL CONFIG ----------")
    if not exists(arg_groups):
        for k, v in vars(args).items():
            print(f"    {k} : {v}")
    else:
        for group in arg_groups:
            print(f"    ---- {group} ----")
            for k, v in vars(arg_groups[group]).items():
                print(f"        {k} : {v}")
