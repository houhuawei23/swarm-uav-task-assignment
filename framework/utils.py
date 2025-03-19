from typing import List, Tuple, Dict
from dataclasses import dataclass, field

import numpy as np
import json
import subprocess

from .uav import UAV, UAVManager
from .task import TaskManager


def get_resources_weights(required_resources, task_obtained_resources):
    still_required_resources = required_resources - task_obtained_resources
    still_required_resources_pos = np.maximum(still_required_resources, 0)  # 将负值置为0
    if np.sum(still_required_resources_pos) == 0:
        return np.zeros_like(still_required_resources_pos)
    else:
        resources_weights = still_required_resources_pos / np.sum(still_required_resources_pos)
        return resources_weights


@dataclass(repr=True)
class EvalResult:
    completion_rate: float = 0.0
    resource_use_rate: float = 0.0
    total_distance: float = 0.0
    total_energy: float = 0.0
    total_exploss: float = 0.0
    elapsed_time: float = 0.0
    # solver_name: str = field(default="")
    # task2coalition: Dict[int | None, List[int]] = field(default=None)

    def format_print(self):
        # print(f"EvalResult for {self.solver_name}")
        print(f" Completion Rate: {self.completion_rate:.2f}")
        print(f" Resource Use Rate: {self.resource_use_rate:.2f}")
        print(f" Total Distance: {self.total_distance:.2f}")
        print(f" Total Energy: {self.total_energy:.2f}")
        print(f" Total Exploss: {self.total_exploss:.2f}")
        print(f" Elapsed Time: {self.elapsed_time:.2f}")
        # print(f"Solver Name: {self.solver_name}")
        # print(f" Task2Coalition: {self.task2coalition}")
        print()

    def to_dict(self):
        return self.__dict__

    def to_flattened_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


def calculate_map_shape(uav_manager: UAVManager, task_manager: TaskManager):
    entities_list = uav_manager.get_all() + task_manager.get_all()
    max_x = max(entity.position.x for entity in entities_list)
    max_y = max(entity.position.y for entity in entities_list)
    max_z = max(entity.position.z for entity in entities_list)

    return (max_x + 1, max_y + 1, max_z + 1)


def calculate_map_shape_beta(uav_list: List[UAV], task_list: List[UAV]):
    entities_list = uav_list + task_list
    max_x = max(entity.position.x for entity in entities_list)
    max_y = max(entity.position.y for entity in entities_list)
    max_z = max(entity.position.z for entity in entities_list)

    return (max_x + 1, max_y + 1, max_z + 1)


def calculate_map_shape_gamma(uav_dict_list: List[UAV], task_sict_list: List[UAV]):
    dict_list = uav_dict_list + task_sict_list
    max_x = max(item["position"][0] for item in dict_list)
    max_y = max(item["position"][1] for item in dict_list)
    max_z = max(item["position"][2] for item in dict_list)

    return (max_x + 1, max_y + 1, max_z + 1)


def calculate_obtained_resources(
    coalition: List[int], uav_manager: UAVManager, resources_num: int
) -> List[float]:
    """
    max O(n)
    """
    obtained_resources = np.zeros(resources_num)
    for uav_id in coalition:
        if uav_id not in uav_manager.get_ids():
            continue
        uav = uav_manager.get(uav_id)
        obtained_resources += uav.resources

    return obtained_resources


def calculate_obtained_resources_beta(uav_coalition: List[UAV], resources_num: int):
    obtained_resources = np.zeros(resources_num)
    for uav in uav_coalition:
        obtained_resources += uav.resources

    return obtained_resources


def calualte_task_completion_rate(
    uav_manager: UAVManager,
    task_manager: TaskManager,
    task2coalition: Dict[int, List[int]],
    resources_num: int,
):
    # 计算任务完成率
    completed_tasks = 0
    task_list = task_manager.get_all()
    for task in task_list:
        obtained = calculate_obtained_resources(task2coalition[task.id], uav_manager, resources_num)
        if np.all(obtained >= task.required_resources):
            completed_tasks += 1

    return completed_tasks / len(task_list)


def calculate_resource_use_rate(
    uav_manager: UAVManager,
    task_manager: TaskManager,
    task2coalition: Dict[int, List[int]],
    resources_num: int,
):
    # 计算资源利用率
    all_input_resources = np.zeros(resources_num)
    unused_resources = np.zeros(resources_num)
    task_list = task_manager.get_all()
    for task in task_list:
        obtained = calculate_obtained_resources(task2coalition[task.id], uav_manager, resources_num)
        all_input_resources += obtained
        unused_resources += np.maximum(obtained - task.required_resources, 0)

    if np.sum(all_input_resources) == 0:
        return 0
    return (1 - np.sum(unused_resources) / np.sum(all_input_resources)).item()


def calculate_total_distance(
    uav_manager: UAVManager, task_manager: TaskManager, task2coalition: Dict[int | None, List[int]]
):
    total_distance = 0.0
    for task_id, coalition in task2coalition.items():
        if task_id is not None:
            task = task_manager.get(task_id)
            uavs = [uav_manager.get(uav_id) for uav_id in coalition]
            for uav in uavs:
                total_distance += uav.position.distance_to(task.position)
    return total_distance


def calculate_total_energy(
    uav_manager: UAVManager, task_manager: TaskManager, task2coalition: Dict[int | None, List[int]]
):
    total_energy = 0.0
    for task_id, coalition in task2coalition.items():
        if task_id is not None:
            task = task_manager.get(task_id)
            uavs = [uav_manager.get(uav_id) for uav_id in coalition]
            # breakpoint()
            for uav in uavs:
                total_energy += (
                    uav.fly_energy_per_time * uav.position.distance_to(task.position)
                    + uav.hover_energy_per_time * task.execution_time
                )
    return total_energy


def calculate_total_exploss(
    uav_manager: UAVManager, task_manager: TaskManager, task2coalition: Dict[int | None, List[int]]
):
    """
    计算期望损失: sum(task.threat * uav.value)
    """
    total_exploss = 0.0
    for task_id, coalition in task2coalition.items():
        if task_id is not None:
            task = task_manager.get(task_id)
            uavs = [uav_manager.get(uav_id) for uav_id in coalition]
            for uav in uavs:
                total_exploss += task.threat * uav.value
    return total_exploss


def evaluate_assignment(
    uav_manager: UAVManager,
    task_manager: TaskManager,
    task2coalition: Dict[int, List[int]],
    resources_num: int,
) -> EvalResult:
    completion_rate = calualte_task_completion_rate(
        uav_manager, task_manager, task2coalition, resources_num
    )
    resource_use_rate = calculate_resource_use_rate(
        uav_manager, task_manager, task2coalition, resources_num
    )
    total_distance = calculate_total_distance(uav_manager, task_manager, task2coalition)
    total_energy = calculate_total_energy(uav_manager, task_manager, task2coalition)
    total_exploss = calculate_total_exploss(uav_manager, task_manager, task2coalition)

    return EvalResult(
        completion_rate, resource_use_rate, total_distance, total_energy, total_exploss
    )


def format_json(json_file_path, config_path=".prettierrc"):
    try:
        # 调用 Prettier 命令行工具
        subprocess.run(["prettier", "--write", json_file_path], check=True)
        print(f"'{json_file_path}' has been formatted with Prettier.")
    except subprocess.CalledProcessError as e:
        print(f"Error formatting JSON file with Prettier: {e}")
    except FileNotFoundError:
        print("Prettier is not installed or not found in your system PATH.")


def save_uavs_and_tasks(uav_manager: UAVManager, task_manager: TaskManager, output_file_path):
    # 将 UAVManager 和 TaskManager 的信息存储为 JSON 文件
    data = {
        "uavs": uav_manager.to_dict_list(),
        "tasks": task_manager.to_dict_list(),
    }

    with open(output_file_path, "w") as f:
        json.dump(data, f)
    format_json(output_file_path)


import pandas as pd


def flatten_dict(base_dict: Dict, parent_key="", sep="_"):
    items = []
    for k, v in base_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(flat_dict, key_prefixes, sep="_"):
    nested_dict = {}

    for key, value in flat_dict.items():
        # 识别 key 属于哪个前缀
        matched_prefix = None
        for prefix in key_prefixes:
            if key.startswith(prefix + sep) or key == prefix:
                matched_prefix = prefix
                break

        if matched_prefix:
            sub_key = (
                key[len(matched_prefix) + 1 :] if key != matched_prefix else None
            )  # 去除前缀部分
            if sub_key:
                current = nested_dict.setdefault(matched_prefix, {})
                current[sub_key] = value
            else:
                nested_dict[matched_prefix] = value  # 直接存储无子 key 的项
        else:
            nested_dict[key] = value  # 非嵌套 key 直接存入

    return nested_dict
