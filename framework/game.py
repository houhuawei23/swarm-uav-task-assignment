from dataclasses import dataclass, field

from .base import HyperParams
from .uav import UAVManager
from .task import TaskManager
from .coalition import CoalitionManager


@dataclass
class CoalitionFormationGame:
    uav_manager: UAVManager
    task_manager: TaskManager
    coalition_manager: CoalitionManager
    hyper_params: HyperParams

    def run_allocate(self, debug=False):
        raise NotImplementedError("This method should be implemented by the subclass.")


if __name__ == "__main__":
    pass
