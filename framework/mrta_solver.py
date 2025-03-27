from typing import Type
from dataclasses import dataclass, field

from .base import HyperParams, LogLevel
from .uav import UAV, UAVManager
from .task import TaskManager
from .coalition_manager import CoalitionManager


# @dataclass
class MRTASolver:
    uav_manager: UAVManager
    task_manager: TaskManager
    coalition_manager: CoalitionManager
    hyper_params: HyperParams

    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
    ):
        self.uav_manager = uav_manager
        self.task_manager = task_manager
        self.coalition_manager = coalition_manager
        self.hyper_params = hyper_params

    @classmethod
    def type_name(cls):
        return cls.__name__

    @staticmethod
    def uav_type() -> Type[UAV]:
        return UAV

    def run_allocate(self):
        iter_cnt = 0

        while True:
            print(f"iter {iter_cnt}")
            if iter_cnt > self.hyper_params.max_iter:
                print(f"reach max iter {self.hyper_params.max_iter}")
                break
            iter_cnt += 1

        raise NotImplementedError("This method should be implemented by the subclass.")


if __name__ == "__main__":
    pass
