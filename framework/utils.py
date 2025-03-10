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
class EvaluationResult:
    completion_rate: float = 0.0
    resource_use_rate: float = 0.0


def calculate_map_shape(uav_manager: UAVManager, task_manager: TaskManager):
    entities_list = uav_manager.get_all() + task_manager.get_all()
    max_x = max(entity.position.x for entity in entities_list)
    max_y = max(entity.position.y for entity in entities_list)

    return (max_x + 1, max_y + 1, 0)


def calculate_obtained_resources(
    coalition: List[int], uav_manager: UAVManager, resources_num: int
) -> List[float]:
    obtained_resources = np.zeros(resources_num)
    for uav_id in coalition:
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


def evaluate_assignment(
    uav_manager: UAVManager,
    task_manager: TaskManager,
    task2coalition: Dict[int, List[int]],
    resources_num: int,
) -> EvaluationResult:
    completion_rate = calualte_task_completion_rate(
        uav_manager, task_manager, task2coalition, resources_num
    )
    resource_use_rate = calculate_resource_use_rate(
        uav_manager, task_manager, task2coalition, resources_num
    )

    return EvaluationResult(completion_rate, resource_use_rate)


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
