import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
from base import Point, Entity, EntityManager


@dataclass(init=True, repr=True)
class Task(Entity):
    """Represents a task that requires the cooperation of UAVs to be completed.

    task.info = (Resources, Position, Time Window, Threat Index)

    Attributes:
        id (int): Unique identifier for the task.
        resources (list or np.ndarray): A vector representing the resources required to complete the task.
        position (tuple or list): A 3D coordinate (x, y, z) representing the task's location.
        time_window (list): A time window [min_start, max_start] during which the task can be started.
        threat (float): A threat index representing the danger or risk associated with the task.
    """

    required_resources: np.ndarray  # 资源需求向量
    time_window: List  # 时间窗口 [min_start, max_start]
    threat: float  # 威胁指数

    resources_nums: int = field(default=0, init=False)  # 资源数量

    def __post_init__(self):
        super().__post_init__()
        self.required_resources = np.array(self.required_resources)
        self.resources_nums = len(self.required_resources)

    def get_resources_weights(self, task_obtained_resources=0):
        still_required_resources = self.required_resources - task_obtained_resources
        still_required_resources_pos = np.maximum(still_required_resources, 0)  # 将负值置为0
        if np.sum(still_required_resources_pos) == 0:
            return np.zeros_like(still_required_resources_pos)
        else:
            resources_weights = still_required_resources_pos / np.sum(still_required_resources_pos)
            return resources_weights

    def to_dict(self):
        return {
            "id": self.id,
            "required_resources": self.required_resources.tolist(),
            "position": self.position.tolist(),
            "time_window": self.time_window,
            "threat": self.threat,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            required_resources=data["required_resources"],
            position=data["position"],
            time_window=data["time_window"],
            threat=data["threat"],
        )

    def brief_info(self) -> str:
        return f"T_{self.id}(req={self.required_resources}, tw={self.time_window}, thr={self.threat})"


@dataclass
class TaskManager(EntityManager):
    def __init__(self, tasks: List[Task] = []):
        super().__init__(tasks)

    def get(self, id) -> Task:
        return super().get(id)

    @classmethod
    def from_dict(cls, data):
        tasks = [Task.from_dict(task_data) for task_data in data]
        return cls(tasks)

    def format_print(self):
        print(f"TaskManager with {len(self)} tasks.")
        for task in self.get_all():
            print(f"  {task}")


# Example usage:
if __name__ == "__main__":
    # Create some tasks
    task1 = Task(
        id=1,
        required_resources=[5, 2, 3],
        position=[10, 20, 0],
        time_window=[0, 100],
        threat=0.5,
    )
    task2 = Task(
        id=2,
        required_resources=[4, 5, 6],
        position=[40, 50, 0],
        time_window=[50, 150],
        threat=0.7,
    )

    # Initialize TaskManager with a list of tasks
    task_manager = TaskManager(tasks=[task1, task2])

    # Add a new task
    task3 = Task(
        id=3,
        required_resources=[7, 8, 9],
        position=[70, 80, 0],
        time_window=[100, 200],
        threat=0.9,
    )
    task_manager.add(task3)

    # Get a task by ID
    print(task_manager.get(2))

    # Delete a task by ID
    task_manager.remove(1)

    # Get all tasks
    print(task_manager.get_all())

    # Update a task
    task2.required_resources = [10, 11, 12]
    # task_manager.update_task(task2)

    # Print all tasks
    print(task_manager.get_all())

    # Automatic detection (will show 3D because of task 3)
    # task_manager.plot_distribution_beta("2d")

    # Clear all tasks
    # task_manager.clear_tasks()
    print(task_manager.get_all())
