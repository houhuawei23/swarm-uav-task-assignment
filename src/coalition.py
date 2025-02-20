from uav import UAV, UAVManager
from task import Task, TaskManager
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt


class CoalitionSet:
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

        if assignment != None:
            self.coalitions = assignment
            assigned_uavs = []
            for task_id, uav_ids in assignment.items():
                for uav_id in uav_ids:
                    assigned_uavs.append(uav_id)
                    self.uav2task[uav_id] = task_id
            self.coalitions[None] = list(
                set(uav_manager.get_uav_ids()) - set(assigned_uavs)
            )
        else:
            self.coalitions: Dict[int, List[int]] = {
                task.id: [] for task in task_manager
            }
            self.coalitions[None] = uav_manager.get_uav_ids()  # 未分配任务的无人机

        # 维护每个任务当前需要的资源向量
        # self.required_resources = {task.id: task.required_resources for task in task_manager}
        self.task_obtained_resources = {
            task.id: np.zeros(task.required_resources.shape) for task in task_manager
        }

    def assign(self, uav: UAV, task: Task):
        """Assigns a UAV to a task, updating the coalitions dictionary."""
        # print(f"Assigning u{uav.id} to t{task.id}")
        if self.uav2task[uav.id] is not None:
            print(
                "Error: UAV {} has already been assigned to task {}".format(
                    uav.id, self.uav2task[uav.id]
                )
            )
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

    def __call__(self, *args, **kwds):
        pass

    def __getitem__(self, key):
        return self.coalitions[key]

    def __str__(self):
        return str(self.coalitions)

    def plot_map(self, output_path=None):
        """Visualizes the UAVs and tasks on a 2D map, with circles indicating UAV coalitions."""
        plt.figure(figsize=(14, 14))
        # plt.figure()
        text_delta = 0.2  # 文本偏移量
        # Plot tasks
        for task in self.task_manager:
            plt.scatter(
                task.position[0],
                task.position[1],
                color="red",
                label=f"Task {task.id}",
                s=200,
                marker="s",
            )
            plt.text(
                task.position[0],
                task.position[1] + text_delta,
                f"{task}",
                fontsize=12,
                ha="center",
            )
        # Plot UAVs and their coalitions
        for task_id, coalition in self.coalitions.items():
            if task_id is not None:
                task = self.task_manager.get_task_by_id(task_id)
                # Draw a circle around UAVs in the same coalition
                if len(coalition) > 1:
                    x_coords = [
                        self.uav_manager.get_uav_by_id(uav_id).position[0]
                        for uav_id in coalition
                    ]
                    y_coords = [
                        self.uav_manager.get_uav_by_id(uav_id).position[1]
                        for uav_id in coalition
                    ]
                    x_coords.append(task.position[0])
                    y_coords.append(task.position[1])
                    # print(x_coords, y_coords)
                    center_x = np.mean(x_coords)
                    center_y = np.mean(y_coords)
                    radius = (
                        max(np.max(x_coords) - center_x, np.max(y_coords) - center_y)
                        + 3
                    )
                    circle = plt.Circle(
                        (center_x, center_y),
                        radius,
                        color="blue",
                        fill=False,
                        linestyle="--",
                    )
                    plt.gca().add_patch(circle)

                # Plot UAVs
                for uav_id in coalition:
                    uav = self.uav_manager.get_uav_by_id(uav_id)
                    plt.scatter(
                        uav.position[0],
                        uav.position[1],
                        color="blue",
                        label=f"UAV {uav.id}",
                        s=100,
                    )
                    plt.text(
                        uav.position[0],
                        uav.position[1] + text_delta,
                        f"{uav}",
                        fontsize=10,
                        ha="center",
                    )

                    # darw an arrow from UAV to task
                    plt.arrow(
                        uav.position[0],
                        uav.position[1],
                        task.position[0] - uav.position[0],
                        task.position[1] - uav.position[1],
                        color="black",
                        head_width=0.5,
                        head_length=0.5,
                    )
        # Plot unassigned UAVs
        for uav_id in self.get_unassigned_uav_ids():
            uav = self.uav_manager.get_uav_by_id(uav_id)
            plt.scatter(
                uav.position[0],
                uav.position[1],
                color="gray",
                label=f"UAV {uav.id} (Unassigned)",
                s=100,
            )
            plt.text(
                uav.position[0],
                uav.position[1] + text_delta,
                f"{uav}",
                fontsize=10,
                ha="center",
            )
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("UAVs and Tasks on Map")
        plt.grid(True)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.legend()
        if output_path:
            plt.savefig(output_path)
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

    coalition_set = CoalitionSet(uav_manager, task_manager, assignment)

    coalition_set.plot_map()
