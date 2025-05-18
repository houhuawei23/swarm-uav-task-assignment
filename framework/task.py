from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

import random
import numpy as np


from .base import Point, Entity, EntityManager, GenParams


default_execution_time = 2


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

    execution_time: float  # 任务执行时间

    def __init__(
        self,
        id: int,
        position: Point,
        required_resources: List,
        time_window: List,
        threat: float,
        execution_time: float = default_execution_time,
    ):
        super().__init__(id, position)
        self.required_resources = np.array(required_resources)
        self.time_window = time_window
        self.threat = threat
        assert execution_time is not None
        self.execution_time = execution_time

    def __post_init__(self):
        super().__post_init__()

    def __eq__(self, other: "Task"):
        return (
            self.id == other.id
            and np.all(self.position.xyz == other.position.xyz)
            and np.all(self.required_resources == other.required_resources)
            and self.time_window == other.time_window
            and self.threat == other.threat
            and self.execution_time == other.execution_time
        )

    def to_dict(self):
        return {
            "id": self.id,
            "required_resources": self.required_resources.tolist(),
            "position": self.position.tolist(),
            "time_window": self.time_window,
            "threat": self.threat,
            "execution_time": self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            id=data["id"],
            required_resources=data["required_resources"],
            position=data["position"],
            time_window=data["time_window"],
            threat=data["threat"],
            execution_time=data.get("execution_time", random.uniform(1, 3)),
        )

    def brief_info(self) -> str:
        return (
            f"T{self.id}(req={self.required_resources}, tw={self.time_window}, thr={self.threat})"
        )

    def is_zero_task(self):
        return self.id == 0


@dataclass
class TaskManager(EntityManager):
    """
    TaskManager is a class that manages a list of tasks.
    """

    free_uav_task_id = 0

    def __init__(self, tasks: List[Task], resources_num: int):
        super().__init__(tasks)
        self.max_execution_time = max(task.execution_time for task in tasks)
        self.min_execution_time = min(task.execution_time for task in tasks)
        # init free_uav_task
        self.get_free_uav_task(resources_num)

    def get(self, id) -> Task:
        return super().get(id)

    def get_all(self) -> List[Task]:
        return super().get_all()

    @classmethod
    def from_dict(cls, data, resources_num: int):
        tasks = [Task.from_dict(task_data) for task_data in data]
        return cls(tasks, resources_num)

    def format_print(self):
        print(f"TaskManager with {len(self)} tasks.")
        for task in self.get_all():
            print(f"  {task}")

    def get_free_uav_task(self, resources_num: int):
        if TaskManager.free_uav_task_id not in self.get_ids():
            free_uav_task = Task(
                id=TaskManager.free_uav_task_id,
                position=Point([0, 0, 0]),
                required_resources=np.zeros(resources_num),
                time_window=[0, 0],
                threat=0,
                execution_time=0,
            )
            self.add(free_uav_task)
        else:
            free_uav_task = self.get(TaskManager.free_uav_task_id)
        if free_uav_task.required_resources.size != resources_num:
            # free_uav_task.required_resources = np.zeros(resources_num)
            raise ValueError(f"free_uav_task.required_resources.size != resources_num")
        return free_uav_task


@dataclass
class TaskGenParams(GenParams):
    required_resources_range: Tuple[float, float] = field(default=(1, 3))
    time_window_ranges: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0, 10), (15, 20)]
    )
    threat_range: Tuple[float, float] = field(default=(0.25, 0.75))
    execution_time_range: Tuple[float, float] = field(default=None)

    @staticmethod
    def from_gen_params(gen_params: GenParams):
        tp = TaskGenParams(
            region_ranges=gen_params.region_ranges,
            resources_num=gen_params.resources_num,
        )
        return tp


precision = 2


# 生成 Task 数据
def generate_task_dict_list(num_tasks: int, params: TaskGenParams = TaskGenParams()):
    tasks = []
    for id in range(1, num_tasks + 1):
        task = {
            "id": id,
            "required_resources": [
                random.randint(*params.required_resources_range)
                for _ in range(params.resources_num)
            ],
            "position": [
                round(random.uniform(*a_range), precision) for a_range in params.region_ranges
            ],
            "time_window": [random.randint(*w_range) for w_range in params.time_window_ranges],
            "threat": round(random.uniform(*params.threat_range), precision),
        }
        if params.execution_time_range is not None:
            task["execution_time"] = random.randint(*params.execution_time_range)

        tasks.append(task)
    return tasks


def generate_task_list(num_tasks: int, params: TaskGenParams = TaskGenParams()):
    return [Task.from_dict(task_data) for task_data in generate_task_dict_list(num_tasks, params)]


# Example usage:
if __name__ == "__main__":
    pass
