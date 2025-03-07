from typing import List, Tuple, Dict
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge

from .base import plot_entities_on_axes, HyperParams
from .uav import UAV, UAVManager
from .task import Task, TaskManager
from .utils import evaluate_assignment


@dataclass
class CoalitionManager:
    task2coalition: Dict[int, List[int]] = field(default_factory=dict, init=False)
    uav2task: Dict[int, int] = field(default_factory=dict, init=False)

    def __init__(self, uav_ids: List[int], task_ids: List[int]):
        # task -> coalition is empty (only None -> all UAV ids)
        self.task2coalition = {task_id: [] for task_id in task_ids}
        self.task2coalition[None] = uav_ids
        # uav -> task is None
        self.uav2task = {uav_id: None for uav_id in uav_ids}

    def update_from_assignment(self, assignment: Dict[int, List[int]], uav_manager: UAVManager):
        # self.task2coalition.clear()
        self.uav2task.clear()

        self.task2coalition = assignment

        assigned_uavs = []
        for task_id, uav_ids in assignment.items():
            if task_id is None:
                print(f"assignment: {assignment}")
                raise Exception("Task id cannot be None in the given assignment")
            for uav_id in uav_ids:
                assigned_uavs.append(uav_id)
                # update uav2task
                self.uav2task[uav_id] = task_id
        # not assigned uavs
        self.task2coalition[None] = list(set(uav_manager.get_ids()) - set(assigned_uavs))

    def assign(self, uav_id: int, task_id: int | None):
        """Assigns a UAV to a task, updating the coalitions dictionary.
        if task is None, unassign the uav.
        """
        if task_id is None:
            print(f"Assigning u{uav_id} to None")
            self.unassign(uav_id)
            return

        # print(f"Assigning u{uav_id} to t{task_id}")
        if self.uav2task[uav_id] is not None:
            print(
                "Error: UAV {} has already been assigned to task {}".format(
                    uav_id, self.uav2task[uav_id]
                )
            )
            raise Exception(
                "UAV {} has already been assigned to task {}".format(uav_id, self.uav2task[uav_id])
            )
        self.task2coalition[task_id].append(uav_id)
        # self.task_obtained_resources[task_id] += uav.resources
        # print(self.coalitions)
        self.task2coalition[None].remove(uav_id)
        self.uav2task[uav_id] = task_id

    def unassign(self, uav_id: int):
        """Unassigns a UAV from its current task, updating the coalitions dictionary."""
        task_id = self.uav2task[uav_id]
        print(f"Unassigning u{uav_id} from t{task_id}")
        if task_id is None:
            print("Warning: UAV {} is not assigned to any task".format(uav_id))
        else:
            self.task2coalition[task_id].remove(uav_id)
            # self.task_obtained_resources[task_id] -= uav.resources
            self.task2coalition[None].append(uav_id)
            self.uav2task[uav_id] = None

    def merge_coalition_manager(self, cmana: "CoalitionManager"):
        for uav_id, task_id in cmana.get_uav2task().items():
            self.assign(uav_id, task_id)

    def get_unassigned_uav_ids(self) -> List[int]:
        return self.task2coalition[None]

    def get_coalition(self, task_id: int) -> List[int]:
        return self.task2coalition[task_id]

    def get_task2coalition(self) -> Dict[int, List[int]]:
        return self.task2coalition

    def get_uav2task(self) -> Dict[int, int]:
        return self.uav2task

    def get_taskid_by_uavid(self, uavid) -> int | None:
        """
        None means the UAV is not assigned to any task.
        """
        return self.uav2task[uavid]

    def __str__(self):
        return str(self.task2coalition)

    def plot_coalition(
        self,
        ax: plt.Axes,
        task_id: int,
        coalition: List[int],
        uav_manager: UAVManager,
        task_manager: TaskManager,
    ):
        text_delta = 0.2  # 文本偏移量
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
    task1 = Task(4, [4, 2], [5, 5, 0], [0, 100], 0.5)
    task2 = Task(5, [3, 3], [15, 15, 0], [0, 100], 0.7)
    tasks = [task1, task2]
    task_manager = TaskManager(tasks)

    assignment = {
        4: [1, 2],
        5: [3],
    }

    coalition_set = CoalitionManager(uav_manager.get_ids(), task_manager.get_ids())

    # coalition_set.plot_map(uav_manager, task_manager)
