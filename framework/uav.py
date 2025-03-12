from typing import List, Dict, Tuple, Type
from dataclasses import dataclass, field

import random
import numpy as np

from .base import Point, Entity, EntityManager, GenParams


@dataclass(init=True, repr=True)
class UAV(Entity):
    """Represents an Unmanned Aerial Vehicle (UAV) with its attributes and capabilities.

    uav.info = (Resource, Position, Value, Max Speed)

    Attributes:
        id (int): Unique identifier for the UAV.
        resources (list or np.ndarray): A vector representing the resources available on the UAV.
        position (tuple or list): A 3D coordinate (x, y, z) representing the UAV's current position.
        value (float): The intrinsic value or importance of the UAV.
        max_speed (float): The maximum speed at which the UAV can travel.
    """

    resources: np.ndarray  # 资源向量
    value: float  # 无人机价值
    max_speed: float  # 最大速度

    mass: float  # 无人机重量
    fly_energy_per_time: float  # 飞行能耗 per time, per mass
    hover_energy_per_time: float  # 悬停能耗 per time, per mass

    def __init__(
        self,
        id: int,
        position: Point | List[float] | np.ndarray,
        resources: List[float],
        value: float,
        max_speed: float,
        mass: float | None = 1.0,
        fly_energy_per_time: float = random.uniform(1, 3),
        hover_energy_per_time: float = random.uniform(1, 3),
    ):
        super().__init__(id, position)
        self.resources = np.array(resources)
        self.value = value
        self.max_speed = max_speed
        self.mass = mass if mass is not None else 1.0
        self.fly_energy_per_time = fly_energy_per_time
        self.hover_energy_per_time = hover_energy_per_time

    def __post_init__(self):
        super().__post_init__()

    def __eq__(self, other: "UAV") -> bool:
        return (
            self.id == other.id
            and np.all(self.position.xyz == other.position.xyz)
            and np.all(self.resources == other.resources)
            and self.value == other.value
            and self.max_speed == other.max_speed
            and self.mass == other.mass
            and self.fly_energy_per_time == other.fly_energy_per_time
            and self.hover_energy_per_time == other.hover_energy_per_time
        )

    def to_dict(self):
        return {
            "id": self.id,
            "resources": self.resources.tolist(),
            "position": self.position.tolist(),
            "value": self.value,
            "max_speed": self.max_speed,
            "mass": self.mass,
            "fly_energy_per_time": self.fly_energy_per_time,
            "hover_energy_per_time": self.hover_energy_per_time,
        }

    @classmethod
    # 从字典中创建一个对象
    def from_dict(cls, data: Dict):
        return cls(
            id=data["id"],
            resources=data["resources"],
            position=data["position"],
            value=data["value"],
            max_speed=data["max_speed"],
            mass=data.get("mass", random.randint(1, 3)),
            fly_energy_per_time=data.get("fly_energy_per_time", random.uniform(1, 3)),
            hover_energy_per_time=data.get("hover_energy_per_time", random.uniform(1, 3)),
        )

    def brief_info(self) -> str:
        return f"U_{self.id}(re={self.resources}, val={self.value}, spd={self.max_speed})"

    def debug_info(self):
        info = f"AU_{self.id}(re={self.resources}, pos={self.position.tolist()}, val={self.value}, spd={self.max_speed})"
        return info

    def cal_fly_energy(self, target_pos: Point) -> float:
        return (
            self.mass
            * (self.position.distance_to(target_pos) / self.max_speed)
            * self.fly_energy_per_time
        )

    def cal_hover_energy(self, hover_time: float) -> float:
        return self.mass * hover_time * self.hover_energy_per_time

    def move_to(self, target_pos: Point, time_step: float = 1.0):
        distance = self.position.distance_to(target_pos)
        if distance > 0:
            direction_vec = self.position.direction_to(target_pos)
            distance_inc = min(self.max_speed * time_step, distance)
            self.position.xyz = self.position.xyz + direction_vec * distance_inc
        else:
            distance_inc = 0
        return distance_inc


@dataclass
class UAVManager(EntityManager):
    def __init__(self, uavs: List[UAV] = []):
        super().__init__(uavs)

    def get(self, id: int) -> UAV:
        return super().get(id)

    def get_all(self) -> List[UAV]:
        return super().get_all()

    def random_one(self) -> UAV:
        return super().random_one()

    @classmethod
    def from_dict(cls, data: List[Dict], UAVType: Type[UAV] = UAV):
        uavs = [UAVType.from_dict(task_data) for task_data in data]
        return cls(uavs)

    def format_print(self):
        print(f"UAVManager with {len(self)} uavs.")
        for uav in self.get_all():
            print(f"  {uav}")


precision = 2


@dataclass
class UAVGenParams(GenParams):
    resources_range: Tuple[float, float] = field(default=(1, 10))
    value_range: Tuple[float, float] = field(default=(1, 10))
    speed_range: Tuple[float, float] = field(default=(1, 10))

    uav_mass_range: Tuple[float, float] = field(default=None)
    fly_energy_per_time_range: Tuple[float, float] = field(default=None)
    hover_energy_per_time_range: Tuple[float, float] = field(default=None)
    comm_bandwidth_range: Tuple[float, float] = field(default=None)
    trans_power_range: Tuple[float, float] = field(default=None)


def generate_uav_dict_list(num_uavs: int, params: UAVGenParams = UAVGenParams()) -> List[Dict]:
    uavs = []
    for id in range(1, num_uavs + 1):
        uav = {
            "id": id,
            "resources": [
                random.randint(*params.resources_range) for _ in range(params.resources_num)
            ],
            "position": [
                round(random.uniform(*a_range), precision) for a_range in params.region_ranges
            ],
            "value": random.randint(*params.value_range),
            "max_speed": random.randint(*params.speed_range),
        }
        if params.uav_mass_range is not None:
            uav["mass"] = random.randint(*params.uav_mass_range)
        if params.fly_energy_per_time_range is not None:
            uav["fly_energy_per_time"] = random.randint(*params.fly_energy_per_time_range)
        if params.hover_energy_per_time_range is not None:
            uav["hover_energy_per_time"] = random.randint(*params.hover_energy_per_time_range)
        uavs.append(uav)
    return uavs


def generate_uav_list(
    num_uavs: int, params: UAVGenParams = UAVGenParams(), UAVType: Type[UAV] = UAV
) -> List[UAV]:
    return [UAVType.from_dict(uav) for uav in generate_uav_dict_list(num_uavs, params)]


if __name__ == "__main__":
    pass
