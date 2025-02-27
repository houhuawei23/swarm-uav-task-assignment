from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

from uav import UAV
from task import Task


from uav import UAVManager
from task import TaskManager

from dataclasses import dataclass, field


@dataclass(repr=True)
class EvaluationResult:
    completion_rate: float = 0.0
    resource_use_rate: float = 0.0


def calculate_map_shape(uav_manager: UAVManager, task_manager: TaskManager):
    entities_list = uav_manager.get_all() + task_manager.get_all()
    max_x = max(entity.position.x for entity in entities_list)
    max_y = max(entity.position.y for entity in entities_list)

    return (max_x + 1, max_y + 1, 0)


def calculate_obtained_resources(collation: List[int], uav_manager: UAVManager, resources_num: int) -> List[float]:
    obtained_resources = np.zeros(resources_num)
    for uav_id in collation:
        uav = uav_manager.get(uav_id)
        obtained_resources += uav.resources

    return obtained_resources


def calualte_task_completion_rate(
    uav_manager: UAVManager, task_manager: TaskManager, task2coalition: Dict[int, List[int]], resources_num: int
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
    uav_manager: UAVManager, task_manager: TaskManager, task2coalition: Dict[int, List[int]], resources_num: int
):
    # 计算资源利用率
    all_input_resources = np.zeros(resources_num)
    unused_resources = np.zeros(resources_num)
    task_list = task_manager.get_all()
    for task in task_list:
        obtained = calculate_obtained_resources(task2coalition[task.id], uav_manager, resources_num)
        all_input_resources += obtained
        unused_resources += np.maximum(obtained - task.required_resources, 0)

    return (1 - np.sum(unused_resources) / np.sum(all_input_resources)).item()


def evaluate_assignment(
    uav_manager: UAVManager, task_manager: TaskManager, task2coalition: Dict[int, List[int]], resources_num: int
) -> EvaluationResult:
    completion_rate = calualte_task_completion_rate(uav_manager, task_manager, task2coalition, resources_num)
    resource_use_rate = calculate_resource_use_rate(uav_manager, task_manager, task2coalition, resources_num)

    return EvaluationResult(completion_rate, resource_use_rate)


def calculate_resource_contribution(
    Kj: np.ndarray,
    I: np.ndarray,
    O: float,
    P: float,
):
    """
    Returns:
        float: The resource contribution value.

    val(ui, tj) = K[j, :] · I - P · O
        K: (m, l) 权重矩阵, P: 常数权值, 对未利用资源的惩罚系数
        K[j, :] = [k1, k2, ..., kl]: 第 j 个任务的资源权重向量, K[j, k] 表示第 k 个资源的权重
        I = [i1, i2, ..., il]: 无人机加入任务对应的联盟集合中可利用的每类资源的数目, 取决于任务本身与联盟内其他成员?
        O: 该无人机未利用的总资源数目, 即加入该联盟后冗余资源总数目
    """
    contribution = np.dot(Kj, I) - P * O
    return max(0, contribution)


def calculate_path_cost(uav: UAV, task: Task, resource_contribution: float, map_shape: Tuple, mu: float = -1.0):
    """Calculates the path cost for a UAV to reach a task.

    The path cost is computed based on the Euclidean distance between the UAV's current position and
    the task's position. The cost is normalized by the maximum possible distance in the environment.
    If the distance exceeds the maximum distance, the UAV is considered unable to reach the task.

    当 val(ui, tj) <= 0 时, 设计 rui (tj) 小于 0.
    含义是当无人机 ui 加入任务 tj 联盟无法 贡献资源时, 加入该联盟的收益小于在 ct0 中的收益

    Args:
        uav (UAV): The UAV object representing the unmanned aerial vehicle.
        task (Task): The task object representing the task to be completed.

    Returns:
        float: The normalized path cost (between 0 and 1) if the task is reachable, otherwise -1.
    """
    # 路径成本计算
    if resource_contribution <= 0:
        return mu

    # distance = np.linalg.norm(uav.position - task.position)
    distance = uav.position.distance_to(task.position)
    # max_distance = np.sqrt(100**2 + 100**2)  # 假设最大距离
    max_distance = np.linalg.norm(map_shape)  # 任务环境区域大小

    return 1 - distance / max_distance


def calculate_threat_cost(uav: UAV, task: Task):
    """Calculates the threat cost for a UAV assigned to a task.

    The threat cost is computed as the product of the UAV's intrinsic value and the task's threat index.
    This represents the risk or potential loss associated with assigning the UAV to the task.

    Args:
        uav (UAV): The UAV object representing the unmanned aerial vehicle.
        task (Task): The task object representing the task to be completed.

    Returns:
        float: The threat cost value.
    """
    # 威胁代价计算
    return uav.value * task.threat


import json
import subprocess


def format_json(json_file_path):
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
        "uavs": uav_manager.to_dict(),
        "tasks": task_manager.to_dict(),
    }

    with open(output_file_path, "w") as f:
        json.dump(data, f)
    format_json(output_file_path)
