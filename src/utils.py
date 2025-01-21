from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

from uav import UAV
from task import Task


from uav import UAVManager
from task import TaskManager
from coalition import CoalitionSet


def calculate_uav_task_benefit(
    uav: UAV,
    task: Task,
    coalition: List[UAV], # UAVs in the coalition, without the current UAV
    map_shape,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    debug=False,
) -> float:
    """
    r(ui, tj) = alpha * val(ui, tj) + beta * cost(ui, tj) - gamma * risk(ui, tj)
    """
    # 计算资源贡献
    Kj = task.resources_weights
    satisfied = np.zeros(resources_num)
    for uav_in_coalition in coalition:
        satisfied += uav_in_coalition.resources
    
    I = uav.resources  # only consider uav's own resources
    # +is required, -is surplus
    pre_required_resources = task.required_resources - satisfied
    # +is required, -is surplus
    now_required_resources = np.maximum(pre_required_resources, 0) - uav.resources
    # +is surplus, -is required
    now_not_required_resources = -now_required_resources

    O = sum(np.maximum(now_not_required_resources, 0))
    P = 0.5
    val = calculate_resource_contribution

    # 计算路径成本
    cost = calculate_path_cost(uav, task, map_shape, val)

    # 计算威胁代价
    risk = calculate_threat_cost(uav, task)

    # 总收益
    total_benefit = alpha * val + beta * cost - gamma * risk

    return total_benefit


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


def calculate_path_cost(uav: UAV, task: Task, map_shape, val, mu=-1.0):
    """Calculates the path cost for a UAV to reach a task.

    The path cost is computed based on the Euclidean distance between the UAV's current position and
    the task's position. The cost is normalized by the maximum possible distance in the environment.
    If the distance exceeds the maximum distance, the UAV is considered unable to reach the task.

    Args:
        uav (UAV): The UAV object representing the unmanned aerial vehicle.
        task (Task): The task object representing the task to be completed.

    Returns:
        float: The normalized path cost (between 0 and 1) if the task is reachable, otherwise -1.
    """
    # 路径成本计算
    if val <= 0:
        return mu

    distance = np.linalg.norm(uav.position - task.position)
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
