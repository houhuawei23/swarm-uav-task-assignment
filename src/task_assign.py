from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import numpy as np
import itertools
from uav import UAV, UAVManager
from task import Task, TaskManager
from coalition import CoalitionSet
import matplotlib.pyplot as plt

from utils import *
from itertools import product
from typing import List, Dict


def generate_all_assignments(
    uav_ids: List[int], task_ids: List[int]
) -> List[Dict[int, List[int]]]:
    """
    Generates all possible assignments of tasks to UAVs.

    Args:
        uav_ids: A list of UAV IDs.
        task_ids: A list of task IDs.

    Returns:
        A list of dictionaries, where each dictionary maps a task ID to a list of UAV IDs assigned to that task.
    """
    assignments = []

    def backtrack(uav_index, current_assignment):
        if uav_index == len(uav_ids):
            # 所有无人机都已处理，将当前分配方案添加到结果中
            # 使用固定的任务 ID 顺序来确保格式一致
            sorted_assignment = {
                task_id: sorted(current_assignment.get(task_id, []))
                for task_id in task_ids
            }
            if sorted_assignment not in assignments:
                assignments.append(sorted_assignment)
            return

        # 当前无人机可以选择不分配给任何任务
        backtrack(uav_index + 1, current_assignment)

        # 当前无人机可以选择分配给某个任务
        for task_id in task_ids:
            if task_id not in current_assignment:
                current_assignment[task_id] = []
            current_assignment[task_id].append(uav_ids[uav_index])
            backtrack(uav_index + 1, current_assignment)
            current_assignment[task_id].pop()  # 回溯

            # 如果当前任务没有无人机分配，则移除该任务
            if not current_assignment[task_id]:
                del current_assignment[task_id]

    # 开始回溯
    backtrack(0, {})

    return assignments


class TaskAssignmentAlgorithm(ABC):
    """
    Abstract base class for task assignment algorithms.

    Attributes:
        uav_manager (UAVManager): The UAVManager instance.
        task_manager (TaskManager): The TaskManager instance.
    """

    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        resources_num: int,
        map_shape,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    ):
        """
        Initializes the task assignment algorithm.

        Args:
            uav_manager: The UAVManager instance.
            task_manager: The TaskManager instance.
        """
        self.uav_manager = uav_manager
        self.task_manager = task_manager
        self.resources_num: int = resources_num  # 资源维度数
        self.map_shape = map_shape  # 任务环境区域大小
        self.alpha: float = alpha  # 资源贡献权重
        self.beta: float = beta  # 路径成本权重
        self.gamma: float = gamma  # 威胁权重

    @abstractmethod
    def solve(self) -> Tuple[Dict[int, List[int]], float]:
        """
        Solves the task assignment problem.

        Returns:
            A tuple containing:
              - A dictionary mapping task IDs to a list of UAV IDs assigned to that task.
              - The objective function value (e.g., total reward, minimized cost).
        """
        pass


def calcualte_assignment_score(
    assignment: Dict[int, List[int]],
    uav_manager: UAVManager,
    task_manager: TaskManager,
    map_shape,
    alpha,
    beta,
    gamma,
):
    score = 0
    for task_id, uav_ids in assignment.items():
        task = task_manager.get_task_by_id(task_id)
        task_score = 0
        all_uav_resources = np.zeros_like(task.required_resources)
        for uav_id in uav_ids:
            uav = uav_manager.get_uav_by_id(uav_id)
            distance = np.linalg.norm(uav.position - task.position)
            max_distance = np.linalg.norm(map_shape)
            resource_contribution = uav.resources.sum()  # simple
            all_uav_resources += uav.resources
            path_cost = 1 - distance / max_distance
            threat_cost = uav.value * task.threat
            task_score += (
                alpha * resource_contribution + beta * path_cost - gamma * threat_cost
            )
        resource_overflow = np.maximum(
            (all_uav_resources - task.required_resources), 0
        ).sum()
        # print(f"resource_overflow: {resource_overflow}")
        task_score -= alpha * resource_overflow
        score += task_score
    return score


class EnumerationAlgorithm(TaskAssignmentAlgorithm):
    """
    Implements an enumeration (brute-force) algorithm for task assignment.
    This algorithm checks all possible combinations of UAVs and tasks.
    """

    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        resources_num: int,
        map_shape,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    ):
        super().__init__(
            uav_manager,
            task_manager,
            resources_num,
            map_shape,
            alpha,
            beta,
            gamma,
        )

    def solve(self) -> Tuple[Dict[int, List[int]], float]:
        """
        Solves the task assignment problem using enumeration.
        """
        # Initialize with negative infinity for maximization

        uav_ids = self.uav_manager.get_uav_ids()
        task_ids = self.task_manager.get_task_ids()
        best_assignment = None
        best_score = -float("inf")
        all_assignments = generate_all_assignments(uav_ids, task_ids)
        print(f"All {len(all_assignments)} assignments:")
        # for assignment in all_assignments:
        #     print(assignment)
        for assignment in all_assignments:
            # print(assignment)
            score = calcualte_assignment_score(
                assignment,
                self.uav_manager,
                self.task_manager,
                self.map_shape,
                self.alpha,
                self.beta,
                self.gamma,
            )
            # print(f"score: {score}")
            if score > best_score:
                best_score = score
                best_assignment = assignment
        return best_assignment, best_score


if __name__ == "__main__":
    resources_num = 2
    map_shape = (20, 20, 0)
    gamma = 0.1

    # 初始化无人机
    uav1 = UAV(1, [5, 3], [0, 0, 0], 10, 20)
    uav2 = UAV(2, [3, 4], [10, 10, 0], 15, 25)
    uav3 = UAV(3, [2, 5], [20, 20, 0], 20, 30)
    uavs = [uav1, uav2, uav3]
    uav_manager = UAVManager(uavs)
    # 初始化任务
    task1 = Task(5, [4, 2], [5, 5, 0], [0, 100], 0.5)
    task2 = Task(7, [3, 5], [15, 15, 0], [0, 100], 0.7)
    tasks = [task1, task2]
    task_manager = TaskManager(tasks)
    enumeration_algorithm = EnumerationAlgorithm(
        uav_manager,
        task_manager,
        resources_num=resources_num,
        map_shape=map_shape,
        gamma=gamma,
    )
    best_assignment, best_score = enumeration_algorithm.solve()

    print(f"Best Assignment: {best_assignment}")
    print(f"Best Score: {best_score}")

    coalition_set = CoalitionSet(uav_manager, task_manager, assignment=best_assignment)
    coalition_set.plot_map()

    # uav_ids = uav_manager.get_uav_ids()
    # task_ids = task_manager.get_task_ids()
    # assignments = generate_all_assignments(uav_ids, task_ids)
    # # print(assignments)
    # for idx, assignment in enumerate(assignments):
    #     print(f"方案 {idx + 1}: {assignment}")
