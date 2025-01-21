from uav import UAV, UAVManager
from task import Task, TaskManager
from typing import List, Tuple, Dict


class CoalitionSet:
    def __init__(self, uav_manager: UAVManager, task_manager: TaskManager):
        # 任务联盟: task.id: int -> uav list: List[UAV]
        self.coalitions: Dict[int, List[UAV]] = {task.id: [] for task in task_manager}
        self.coalitions[None] = uav_manager.get_all_uavs()  # 未分配任务的无人机
        self.uav2task = {uav.id: None for uav in uav_manager}  # 无人机-任务对应关系

    def assign(self, uav: UAV, task: Task):
        """Assigns a UAV to a task, updating the coalitions dictionary."""
        if self.uav2task[uav.id] is not None:
            print(
                "Error: UAV {} has already been assigned to task {}".format(
                    uav.id, self.uav2task[uav.id]
                )
            )
            return
            # self.coalitions[self.uav2task[uav.id]].remove(uav)
        self.coalitions[task.id].append(uav)
        self.coalitions[None].remove(uav)
        self.uav2task[uav.id] = task.id

    def unassign(self, uav: UAV):
        """Unassigns a UAV from its current task, updating the coalitions dictionary."""
        if self.uav2task[uav.id] is None:
            print("Error: UAV {} is not assigned to any task".format(uav.id))
        else:
            self.coalitions[self.uav2task[uav.id]].remove(uav)
            self.coalitions[None].append(uav)
            self.uav2task[uav.id] = None

    def get_unassigned_uavs(self):
        return self.coalitions[None]

    def get_coalition(self, task_id):
        return self.coalitions[task_id]

    def get_task_by_uav(self, uav_id):
        return self.uav2task[uav_id]

    def __call__(self, *args, **kwds):
        pass

    def __getitem__(self, key):
        return self.coalitions[key]

    def __str__(self):
        return str(self.coalitions)
