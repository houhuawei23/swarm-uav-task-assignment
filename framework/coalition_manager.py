from typing import List, Tuple, Dict
from dataclasses import dataclass, field
import warnings
import logging

# from copy import deepcopy
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge

from .base import plot_entities_on_axes, HyperParams
from .uav import UAV, UAVManager
from .task import Task, TaskManager
from .utils import evaluate_assignment


class CoalitionManager:
    # task2coalition: Dict[int, List[int]] = field(default_factory=dict, init=False)
    # uav2task: Dict[int, int] = field(default_factory=dict, init=False)

    def __init__(self, uav_ids: List[int], task_ids: List[int]):
        # task -> coalition is empty (only None -> all UAV ids)
        self.task2coalition = {task_id: [] for task_id in task_ids}
        self.task2coalition[None] = copy.deepcopy(uav_ids)
        # uav -> task is None
        self.uav2task = {uav_id: None for uav_id in uav_ids}

    def __str__(self):
        return str(self.task2coalition)

    def deepcopy(self):
        # copy = CoalitionManager([], [])
        # copy.task2coalition = copy.deepcopy(self.task2coalition)
        # copy.uav2task = copy.deepcopy(self.uav2task)
        # return copy
        return copy.deepcopy(self)

    def update_from_assignment(self, assignment: Dict[int, List[int]], uav_manager: UAVManager):
        self.task2coalition = copy.deepcopy(assignment)
        # self.task2coalition = assignment
        self.uav2task.clear()

        for task_id, uav_ids in assignment.items():
            for uav_id in uav_ids:
                self.uav2task[uav_id] = task_id

    def assign(self, uav_id: int, task_id: int | None):
        """Assigns a UAV to a task, updating the coalitions dictionary.
        if task is None, unassign the uav.
        """
        if task_id is None:
            # print(f"Assigning u{uav_id} to None")
            self.unassign(uav_id)
            return

        # print(f"Assigning u{uav_id} to t{task_id}")
        if self.uav2task[uav_id] is not None:
            self.unassign(uav_id)

            # raise Exception(
            #     f"UAV {uav_id} has already been assigned to task {self.uav2task[uav_id]}"
            # )
        self.task2coalition[task_id].append(uav_id)
        self.task2coalition[None].remove(uav_id)
        self.uav2task[uav_id] = task_id

    def unassign(self, uav_id: int):
        """Unassigns a UAV from its current task, updating the coalitions dictionary."""
        task_id = self.uav2task[uav_id]
        # print(f"Unassigning u{uav_id} from t{task_id}")
        if task_id is None:
            # print(f"Warning: UAV {uav_id} is not assigned to any task")
            # warnings.warn(f"Warning: UAV {uav_id} is not assigned to any task", UserWarning)
            pass
        else:
            self.task2coalition[task_id].remove(uav_id)
            self.task2coalition[None].append(uav_id)
            self.uav2task[uav_id] = None

    def merge_coalition_manager(self, cmana: "CoalitionManager"):
        """
        1. cmana.uav2task[uavid] == self.uav2task[uav_id]: pass
        2. cmana.uav2task[uavid] != self.uav2task[uav_id]:
            2.1 cmana.uav2task[uavid] is None, self.uav2task[uav_id] is not None
            2.2 cmana.uav2task[uavid] is not None, self.uav2task[uav_id] is None
            2.3 cmana.uav2task[uavid] is not None, self.uav2task[uav_id] is not None

        """
        # print(f"self: {self.uav2task}")
        # print(f"self: {self.task2coalition}")
        # print(f"cmana: {cmana.uav2task}")
        # print(f"cmana: {cmana.task2coalition}")
        for uav_id, task_id in cmana.get_uav2task().items():
            # if task_id is None and self.uav2task[uav_id] is not None:
            if task_id == self.uav2task[uav_id]:
                continue
            # task_id != self.uav2task[uav_id]
            if (task_id is None) and (self.uav2task[uav_id] is not None):
                # self.assign(uav_id, task_id)
                continue
            elif (task_id is not None) and (self.uav2task[uav_id] is None):
                self.assign(uav_id, task_id)
            else:  # (task_id is not None) and (self.uav2task[uav_id] is not None):
                raise Exception("Cannot merge coalition managers with conflicting assignments")

    def get_unassigned_uav_ids(self) -> List[int]:
        return self.task2coalition[None]

    def get_coalition(self, task_id: int) -> List[int]:
        return self.task2coalition[task_id]

    def get_coalition_by_uav_id(self, uav_id: int) -> List[int]:
        return self.task2coalition[self.uav2task[uav_id]]

    def get_taskid(self, uavid) -> int | None:
        """
        None means the UAV is not assigned to any task.
        """
        # assert uavid in self.uav2task.keys()
        if uavid not in self.uav2task.keys():
            raise Exception(f"UAV {uavid} is not in the coalition manager")
        return self.uav2task[uavid]

    def get_task2coalition(self) -> Dict[int, List[int]]:
        return self.task2coalition

    def get_uav2task(self) -> Dict[int, int]:
        return self.uav2task

    def format_print(self):
        print(f"task2coalition: {self.task2coalition}")
        print(f"uav2task: {self.uav2task}")

    def brief_info(self):
        info = f"task2coalition: {self.task2coalition}, "
        info += f"uav2task: {self.uav2task}"
        return info

    def plot_coalition(
        self,
        ax: plt.Axes,
        task_id: int,
        coalition: List[int],
        uav_manager: UAVManager,
        task_manager: TaskManager,
    ):
        task = task_manager.get(task_id)
        # Draw a circle around UAVs in the same coalition
        if len(coalition) > 1:
            x_coords = [uav_manager.get(uav_id).position.x for uav_id in coalition]
            y_coords = [uav_manager.get(uav_id).position.y for uav_id in coalition]
            x_coords.append(task.position.x)
            y_coords.append(task.position.y)
            # print(x_coords, y_coords)
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            radius = max(np.max(x_coords) - center_x, np.max(y_coords) - center_y) + 3

            circle = Circle(
                (center_x, center_y),
                radius,
                color="blue",
                fill=False,
                linestyle="--",
            )
            ax.add_patch(circle)

        # Plot UAVs
        uav_list = [uav_manager.get(uav_id) for uav_id in coalition]
        plot_entities_on_axes(ax, uav_list, color="blue", marker="o")

        # darw an arrow from UAV to task
        for uav_id in coalition:
            uav = uav_manager.get(uav_id)
            delta_xyz = task.position.xyz - uav.position.xyz
            unit_delta_xyz = delta_xyz / np.linalg.norm(delta_xyz)
            arrow_x, arrow_y, _ = uav.position.xyz + unit_delta_xyz * 0.5
            arrow_dx, arrow_dy, _ = delta_xyz - unit_delta_xyz * 1.5
            ax.arrow(
                arrow_x,
                arrow_y,
                arrow_dx,
                arrow_dy,
                color="black",
                head_width=0.5,
                head_length=0.5,
            )

    def plot_map(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        hyper_params: HyperParams = None,
        output_path=None,
        plot_unassigned=True,
        show=True,
    ):
        fig, ax = plt.subplots(figsize=(14, 14))

        # Plot tasks
        task_manager.plot(ax, color="red", marker="s")
        # uav_manager.plot(ax, color="blue", marker="o")

        # Plot UAVs and their coalitions
        for task_id, coalition in self.task2coalition.items():
            if task_id is None:
                continue
            self.plot_coalition(ax, task_id, coalition, uav_manager, task_manager)

        # Plot unassigned UAVs
        if plot_unassigned:
            unassigned_uavs = [uav_manager.get(uav_id) for uav_id in self.get_unassigned_uav_ids()]
            plot_entities_on_axes(ax, unassigned_uavs, color="gray", marker="o")

        # evaluate and add text
        if hyper_params:
            resources_num = hyper_params.resources_num
        else:
            resources_num = len(uav_manager.random_one().resources)

        eval_result = evaluate_assignment(
            uav_manager,
            task_manager,
            self.task2coalition,
            resources_num,
        )

        ax.text(
            0.75,
            0.95,
            f"Task completion rate: {eval_result.completion_rate:.2f}\nResource use rate: {eval_result.resource_use_rate:.2f}",
            transform=ax.transAxes,
            fontsize=12,
            ha="center",
            va="center",
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Entities")
        ax.grid(True)
        ax.legend()
        if output_path:
            plt.savefig(output_path)

        if show:
            plt.show()


if __name__ == "__main__":
    pass
