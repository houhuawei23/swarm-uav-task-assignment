from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import numpy as np

from framework.base import HyperParams
from framework.uav import UAV, UAVManager
from framework.task import Task, TaskManager
from framework.coalition_manager import CoalitionManager
from framework.mrta_solver import MRTASolver
from framework.utils import calculate_obtained_resources

def generate_all_assignments(uav_ids: List[int], task_ids: List[int]) -> List[Dict[int, List[int]]]:
    """
    Generates all possible assignments of tasks to UAVs.

    Args:
        uav_ids: A list of UAV IDs.
        task_ids: A list of task IDs.

    Returns:
        A list of dictionaries, where each dictionary maps a task ID to a list of UAV IDs assigned to that task.
    """
    assignments = []

    def backtrack(uav_index: int, current_assignment: Dict[int, List[int]]):
        if uav_index == len(uav_ids):
            # 所有无人机都已处理，将当前分配方案添加到结果中
            # 使用固定的任务 ID 顺序来确保格式一致
            sorted_assignment = {
                task_id: sorted(current_assignment.get(task_id, [])) for task_id in task_ids
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


from . import csci2024


def calcualte_assignment_score(
    assignment: Dict[int, List[int]],
    uav_manager: UAVManager,
    task_manager: TaskManager,
    hyper_params: HyperParams,
):
    score = 0
    for task_id, uav_ids in assignment.items():
        task = task_manager.get(task_id)
        task_score = 0
        coalition = assignment[task_id]
        for uav_id in uav_ids:
            uav = uav_manager.get(uav_id)
            obtained_resources = calculate_obtained_resources(
                coalition, uav_manager, hyper_params.resources_num
            )
            benefit = csci2024.calculate_uav_benefit_4_join_task_coalition(
                uav, task, coalition, obtained_resources, hyper_params
            )
            task_score += benefit
        score += task_score
    return score


class EnumerationSolver(MRTASolver):
    """
    Implements an enumeration (brute-force) algorithm for task assignment.
    This algorithm checks all possible combinations of UAVs and tasks.
    """

    def run_allocate(self, debug=False):
        """
        Solves the task assignment problem using enumeration.
        """
        # Initialize with negative infinity for maximization

        uav_ids = self.uav_manager.get_ids()
        task_ids = self.task_manager.get_ids()
        best_assignment = None
        best_score = -float("inf")
        all_assignments = generate_all_assignments(uav_ids, task_ids)
        print(f"All {len(all_assignments)} assignments:")
        # for assignment in all_assignments:
        #     print(assignment)
        for assignment in all_assignments:
            score = calcualte_assignment_score(
                assignment, self.uav_manager, self.task_manager, self.hyper_params
            )
            print(f"{assignment}; score: {score: .3f}")
            # print(f"score: {score}")
            if score > best_score:
                best_score = score
                best_assignment = assignment
        print(f"Best Assignment: {best_assignment}")
        self.coalition_manager.update_from_assignment(best_assignment)


if __name__ == "__main__":
    pass
