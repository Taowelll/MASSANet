from pathlib import Path

import torch
import yaml
from easydict import EasyDict


def print_log(inputs: str):
    print(f"LOG >>> {inputs}")


def normalize_config_name(config_name: str) -> str:
    return config_name[:-5] if config_name.endswith(".yaml") else config_name


def load_config(config_name: str) -> EasyDict:
    normalized_name = normalize_config_name(config_name)
    config_path = Path("configs") / f"{normalized_name}.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        args = EasyDict(yaml.load(file, Loader=yaml.FullLoader))

    args["config_name"] = normalized_name
    return args


def clone_args(args: EasyDict, **updates) -> EasyDict:
    cloned = EasyDict(dict(args))
    cloned.update(updates)
    return cloned


def get_subject_tag(target_subject: int) -> str:
    return f"S{int(target_subject):02d}"


def get_subject_label(target_subject: int) -> str:
    return f"S{int(target_subject) + 1:02d}"


def get_device(gpu_num: str) -> torch.device:
    if torch.cuda.device_count() == 1:
        output = torch.device("cuda")
    elif torch.cuda.device_count() > 1:
        output = torch.device(f"cuda:{gpu_num}")
    else:
        output = torch.device("cpu")

    print_log(f"{output} is checked")
    return output


def get_log_name(args):
    log_list = {
        "time": args.current_time,
        "task": args.task,
        "batch": args.batch_size,
        "lr": args.lr,
        "window": args.window_length,
        "etc": args.log_etc,
    }

    output = log_list["time"]

    for key, value in log_list.items():
        if key in {"time", "etc"}:
            continue
        output += f"_{key}_{value}"

    if log_list["etc"] is not None:
        output += f"_{log_list['etc']}"

    print_log(f"Log name: \n\t{output}")
    return output
