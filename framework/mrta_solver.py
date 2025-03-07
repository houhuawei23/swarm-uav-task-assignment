from dataclasses import dataclass, field

from .base import HyperParams
from .uav import UAVManager
from .task import TaskManager
from .coalition import CoalitionManager


@dataclass
class MRTASolver:
    uav_manager: UAVManager
    task_manager: TaskManager
    coalition_manager: CoalitionManager
    hyper_params: HyperParams

    def run_allocate(self, debug=False):
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
