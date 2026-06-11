"""
# A python module that manipulates torch checkpoint file in a hacky way.
Each function should be used with caution and should be used only when thoughtfully considered.
---
Args:
    source_state_dict: the state_dict loaded using torch.load
    algo_state_dict: the algorithm state_dict summarized from algorithm as an example
---
Returns:
    new_state_dict: the state_dict that has been manipulated or directly saved as a checkpoint file.
"""

import os.path as osp
from collections import OrderedDict
from typing import Literal

import regex as re
import torch


def replace_encoder0(source_state_dict, algo_state_dict):
    print("\033[1;36m Replacing encoder.0 weights with untrained weights and avoid critic_encoder.0 \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if "critic_encoders.0" in key:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
        elif "encoders.0" in key:
            print(
                "key:", key, "shape:", algo_state_dict["model_state_dict"][key].shape, "using untrained module weights."
            )
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    return new_state_dict


def append_GRU_weights(source_state_dict, algo_state_dict):
    print("\033[1;36m Appending GRU weights to fit the new model \033[0m")
    print("\033[1;36m Operating on both actor and critic \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if ("memory_a" in key or "memory_c" in key) and "rnn" in key and "weight_ih" in key:
            print(
                "key:",
                key,
                "shape:",
                source_state_dict["model_state_dict"][key].shape,
                "is updated to shape:",
                algo_state_dict["model_state_dict"][key].shape,
            )
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
            new_model_state_dict[key][:, : source_state_dict["model_state_dict"][key].shape[1]] = source_state_dict[
                "model_state_dict"
            ][key]
            new_model_state_dict[key][:, source_state_dict["model_state_dict"][key].shape[1] :] /= 10
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    return new_state_dict


def append_GRU_weights_newStd(source_state_dict, algo_state_dict):
    return_ = append_GRU_weights(source_state_dict, algo_state_dict)
    print(
        "\033[1;36m Setting the std of the new actor to {} \033[0m".format(
            algo_state_dict["model_state_dict"]["std"].mean().cpu().item()
        )
    )
    return_["model_state_dict"]["std"][:] = algo_state_dict["model_state_dict"]["std"][:]
    return return_


def reinitialize_actor_critic_backbone(source_state_dict, algo_state_dict):
    print("\033[1;36m Reinitializing the actor and critic backbone \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if (
            "actor." in key
            or "critic." in key
            or "critics." in key
            or "memory_a" in key
            or "memory_c" in key
            or "std" in key
        ):
            if not key in source_state_dict["model_state_dict"]:
                print(
                    "key:",
                    key,
                    "shape:",
                    algo_state_dict["model_state_dict"][key].shape,
                    "using untrained module weights.",
                )
            else:
                print(
                    "key:",
                    key,
                    "shape:",
                    source_state_dict["model_state_dict"][key].shape,
                    "is updated to shape:",
                    algo_state_dict["model_state_dict"][key].shape,
                )
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    return new_state_dict


def ignore_missing_key(source_state_dict, algo_state_dict):
    """Ignore the missing critic mlp weights and use the initialized ones."""
    print("\033[1;36m Ignoring missing key and using the initialized weights \033[0m")
    new_model_state_dict = OrderedDict()
    missing_keys = []
    for key in algo_state_dict["model_state_dict"].keys():
        if key in source_state_dict["model_state_dict"]:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
        else:
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
            missing_keys.append(key)
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No optimizer_state_dict
        iter=source_state_dict["iter"],
        infos=source_state_dict["infos"],
    )
    print("\033[1;36m Missing keys: \033[0m", missing_keys)
    return new_state_dict


def fit_smaller_weight(
    source_state_dict: dict,
    algo_state_dict: dict,
    weight_name_regex: str = ".*",
    weight_match_mode: Literal["start", "end"] = "start",
):
    """To fix the weight matrix in algo_state_dict which is smaller than the one in source_state_dict,
    we will copy the part of the weight matrix from source_state_dict to algo_state_dict.
    ## Args:
        weight_name_regex: str
            The regex to match the weight name in algo_state_dict.
        weight_match_mode: Literal["start", "end"]
            If "start", weight_algo = weight_source[:weight_algo.shape[0], :weight_algo.shape[1]]
            If "end", weight_algo = weight_source[-weight_algo.shape[0]:, -weight_algo.shape[1]:]
    """
    print("\033[1;36m Fitting smaller weight matrix, matching \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if re.match(weight_name_regex, key):
            weight_algo = algo_state_dict["model_state_dict"][key]
            weight_source = source_state_dict["model_state_dict"][key]
            if weight_match_mode == "start":
                new_model_state_dict[key] = weight_source[: weight_algo.shape[0], : weight_algo.shape[1]]
            elif weight_match_mode == "end":
                new_model_state_dict[key] = weight_source[-weight_algo.shape[0] :, -weight_algo.shape[1] :]
            else:
                raise ValueError(f"Invalid weight_match_mode: {weight_match_mode}. Must be one of ['start', 'end'].")
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
    )
    for k in source_state_dict.keys():
        if k not in new_state_dict and not k.startswith("optimizer_state_dict"):
            new_state_dict[k] = source_state_dict[k]
    return new_state_dict


def merge_student_actor_teacher_critic(
    source_state_dict: dict,
    algo_state_dict: dict,
    teacher_ckpt_path: "str | None" = None,
    copy_discriminator_optimizer: bool = True,
    std_from_teacher: bool = True,
    reset_iter: bool = True,
):
    """Assemble a finetune checkpoint for a distilled (depth) student.

    Combines two trained checkpoints into one model that matches a freshly-built
    WasabiPPO (depth actor + height-scan critic + discriminator):
      - actor / actor encoders (depth)                <- student (source_state_dict)
      - critic / critic encoders (height-scan)        <- teacher (teacher_ckpt_path)
      - discriminator (+ its optimizer, optional)     <- teacher
      - action std                                    <- teacher by default (see std_from_teacher)

    The main PPO optimizer state is intentionally dropped so finetuning starts with a
    fresh optimizer (the model is a student/teacher hybrid that no saved optimizer
    matches). The discriminator is taken verbatim from the teacher, so its optimizer
    state is consistent and is copied by default to keep Adam momentum warm.

    NOTE: the finetune env's critic obs MUST match the teacher's critic obs (i.e. include
    `base_lin_vel` as the first term -> 920-dim), otherwise the teacher critic weights
    will not match the freshly-built critic and this function raises a shape error.

    ## Args:
        teacher_ckpt_path: path to the teacher (WasabiPPO) checkpoint, e.g.
            "~/Data/20260603_112548/model_39000.pt". Required.
        copy_discriminator_optimizer: copy the teacher's discriminator optimizer state.
        std_from_teacher: take the per-joint action std from the teacher. During pure
            distillation (using_ppo=False) the student's std is never trained and stays at its
            (small) init value, while the teacher's std is the exploration level PPO converged
            to on this very task — a much better starting point for RL finetuning.
        reset_iter: start the finetune iteration counter at 0 (else keep the student's).
    """
    assert teacher_ckpt_path is not None, "merge_student_actor_teacher_critic requires teacher_ckpt_path"
    teacher_ckpt_path = osp.expanduser(teacher_ckpt_path)
    print(
        "\033[1;36m Merging student actor + teacher critic/discriminator. Teacher: {} \033[0m".format(teacher_ckpt_path)
    )
    teacher_state_dict = torch.load(teacher_ckpt_path, map_location="cpu", weights_only=False)

    student_model = source_state_dict["model_state_dict"]
    teacher_model = teacher_state_dict["model_state_dict"]
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if key.startswith("critic.") or key.startswith("critic_encoders."):
            src, origin = teacher_model, "teacher"
        elif key == "std" and std_from_teacher:
            src, origin = teacher_model, "teacher"
        else:
            # actor.*, encoders.* (depth) and any other actor-side weights <- student
            src, origin = student_model, "student"
        if key not in src:
            raise KeyError(f"key '{key}' not found in {origin} checkpoint while merging")
        target_shape = algo_state_dict["model_state_dict"][key].shape
        if src[key].shape != target_shape:
            raise ValueError(
                f"shape mismatch for '{key}' from {origin}: {tuple(src[key].shape)} vs expected "
                f"{tuple(target_shape)}. Check that the finetune critic obs matches the teacher."
            )
        new_model_state_dict[key] = src[key]

    new_state_dict = dict(
        model_state_dict=new_model_state_dict,
        # No PPO optimizer_state_dict -> fresh optimizer for finetuning.
        iter=0 if reset_iter else source_state_dict.get("iter", 0),
        infos=source_state_dict.get("infos", None),
    )
    if "discriminator" not in teacher_state_dict:
        raise KeyError("teacher checkpoint has no 'discriminator'; is it a Wasabi/AMP checkpoint?")
    new_state_dict["discriminator"] = teacher_state_dict["discriminator"]
    if copy_discriminator_optimizer and "discriminator_optimizer" in teacher_state_dict:
        new_state_dict["discriminator_optimizer"] = teacher_state_dict["discriminator_optimizer"]
        print("\033[1;36m Copied teacher discriminator_optimizer state. \033[0m")
    return new_state_dict


def newStd(
    source_state_dict: dict,
    algo_state_dict: dict,
):
    """Replicate everything except for policy std"""
    print(
        "\033[1;36m Setting the std of the new actor to {} \033[0m".format(
            algo_state_dict["model_state_dict"]["std"].mean().cpu().item()
        )
    )
    new_state_dict = OrderedDict()
    for state_dict_key in source_state_dict.keys():
        if state_dict_key == "model_state_dict":
            new_state_dict[state_dict_key] = OrderedDict()
            for model_state_dict_key in source_state_dict[state_dict_key].keys():
                if "std" == model_state_dict_key:
                    new_state_dict[state_dict_key][model_state_dict_key] = algo_state_dict["model_state_dict"][
                        model_state_dict_key
                    ]
                else:
                    new_state_dict[state_dict_key][model_state_dict_key] = source_state_dict[state_dict_key][
                        model_state_dict_key
                    ]
        else:
            new_state_dict[state_dict_key] = source_state_dict[state_dict_key]
    return new_state_dict
