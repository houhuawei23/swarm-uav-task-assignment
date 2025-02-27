from uav import UAV, UAVManager
from task import Task, TaskManager
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge
from base import plot_entities_on_axes


class CoalitionManager:
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        assignment: Dict[int, List[int]] = None,
    ):
        # 任务联盟: task.id: int -> uav list: List[UAV]
        self.uav_manager = uav_manager
        self.task_manager = task_manager
        # 无人机-任务对应关系
        self.uav2task: Dict[int, int] = {uav.id: None for uav in uav_manager}

        if assignment is not None:
            self.coalitions = assignment
            assigned_uavs = []
            for task_id, uav_ids in assignment.items():
                for uav_id in uav_ids:
                    assigned_uavs.append(uav_id)
                    self.uav2task[uav_id] = task_id
            self.coalitions[None] = list(set(uav_manager.get_ids()) - set(assigned_uavs))
        else:
            self.coalitions: Dict[int, List[int]] = {task.id: [] for task in task_manager}
            self.coalitions[None] = uav_manager.get_ids()  # 未分配任务的无人机

        # 维护每个任务当前需要的资源向量
        # self.required_resources = {task.id: task.required_resources for task in task_manager}
        self.task_obtained_resources = {task.id: np.zeros(task.required_resources.shape) for task in task_manager}

    def assign(self, uav: UAV, task: Task):
        """Assigns a UAV to a task, updating the coalitions dictionary."""
        # print(f"Assigning u{uav.id} to t{task.id}")
        if self.uav2task[uav.id] is not None:
            print("Error: UAV {} has already been assigned to task {}".format(uav.id, self.uav2task[uav.id]))
            return
            # self.coalitions[self.uav2task[uav.id]].remove(uav)
        self.coalitions[task.id].append(uav.id)
        self.task_obtained_resources[task.id] += uav.resources
        # print(self.coalitions)
        self.coalitions[None].remove(uav.id)
        self.uav2task[uav.id] = task.id

    def unassign(self, uav: UAV):
        """Unassigns a UAV from its current task, updating the coalitions dictionary."""
        task_id = self.uav2task[uav.id]
        print(f"Unassigning u{uav.id} from t{task_id}")
        if task_id is None:
            print("Error: UAV {} is not assigned to any task".format(uav.id))
        else:
            self.coalitions[task_id].remove(uav.id)
            self.task_obtained_resources[task_id] -= uav.resources
            self.coalitions[None].append(uav.id)
            self.uav2task[uav.id] = None

    def get_unassigned_uav_ids(self):
        return self.coalitions[None]

    def get_coalition(self, task_id):
        return self.coalitions[task_id]

    def get_taskid_by_uavid(self, uavid) -> int:
        return self.uav2task[uavid]

    def __str__(self):
        return str(self.coalitions)

    def plot_coalition(self, ax: plt.Axes, task_id: int, coalition: List[int]):
        text_delta = 0.2  # 文本偏移量
        task = self.task_manager.get(task_id)
        # Draw a circle around UAVs in the same coalition
        if len(coalition) > 1:
            x_coords = [self.uav_manager.get(uav_id).position.x for uav_id in coalition]
            y_coords = [self.uav_manager.get(uav_id).position.y for uav_id in coalition]
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
        uav_list = [self.uav_manager.get(uav_id) for uav_id in coalition]
        plot_entities_on_axes(ax, uav_list, color="blue", marker="o")

        # darw an arrow from UAV to task
        for uav_id in coalition:
            uav = self.uav_manager.get(uav_id)
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

    def plot_map(self, output_path=None, plot_unassigned=True, show=False):
        fig, ax = plt.subplots(figsize=(14, 14))

        # Plot tasks
        self.task_manager.plot(ax, color="red", marker="s")
        # self.uav_manager.plot(ax, color="blue", marker="o")

        # Plot UAVs and their coalitions
        for task_id, coalition in self.coalitions.items():
            if task_id is None:
                continue
            self.plot_coalition(ax, task_id, coalition)

        # Plot unassigned UAVs
        if plot_unassigned:
            unassigned_uavs = [self.uav_manager.get(uav_id) for uav_id in self.get_unassigned_uav_ids()]
            plot_entities_on_axes(ax, unassigned_uavs, color="gray", marker="o")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Entities")
        ax.grid(True)
        ax.legend()

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

    coalition_set = CoalitionManager(uav_manager, task_manager, assignment)

    coalition_set.plot_map()
