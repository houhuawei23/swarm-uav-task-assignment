from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt
import random


class LogLevel(IntEnum):
    SILENCE = 0
    INFO = 1
    DEBUG = 2


@dataclass(repr=True, eq=True)
class Point:
    xyz: np.ndarray

    def __init__(self, xyz: List[float] | np.ndarray | "Point"):
        if isinstance(xyz, Point):
            self.xyz = xyz.xyz
        elif isinstance(xyz, (list, np.ndarray)):
            assert len(xyz) == 3
            self.xyz = np.array(xyz)
        else:
            raise TypeError("xyz must be a list, numpy array, or Point")

    @property
    def x(self) -> float:
        return self.xyz[0]

    @property
    def y(self) -> float:
        return self.xyz[1]

    @property
    def z(self) -> float:
        return self.xyz[2]

    def __eq__(self, other: "Point") -> bool:
        return np.all(self.xyz == other.xyz)

    def __add__(self, other: "Point") -> "Point":
        return Point(self.xyz + other.xyz)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.xyz - other.xyz)

    def distance_to(self, other: "Point") -> float:
        return np.linalg.norm(self.xyz - other.xyz).item()

    def direction_to(self, other: "Point") -> np.ndarray:
        delta = other.xyz - self.xyz
        return delta / np.linalg.norm(delta)

    def tolist(self) -> List[float]:
        return self.xyz.tolist()


@dataclass
class Entity:
    """Base class for all entities in the simulation."""

    id: int
    position: Point

    def __init__(self, id: int, position: Point | List[float] | np.ndarray):
        self.id = id
        self.position = Point(position)
        if not isinstance(self.position, Point):
            raise TypeError("position must be a Point")
        # self.position = Point(position) if isinstance(position, (list, np.ndarray)) else position

    def __post_init__(self):
        if not isinstance(self.id, int):
            raise TypeError("id must be an integer")
        if not isinstance(self.position, Point):
            raise TypeError("position must be a Point")

    @classmethod
    def type_name(cls):
        return cls.__name__

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type_name(),
            "position": self.position.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Entity":
        return cls(data["id"], data["position"])

    def brief_info(self) -> str:
        return f"{self.type_name()} {self.id} at {self.position}"


def plot_entities_on_axes(ax: plt.Axes, entities: List[Entity], **kwargs) -> None:
    text_delta = 0.2  # 文本偏移量
    for entity in entities:
        x, y, z = entity.position.xyz
        ax.scatter(x, y, label=f"{entity.type_name()} {entity.id}", **kwargs)
        ax.text(x, y + text_delta, s=entity.brief_info(), ha="center", fontsize=12)


@dataclass
class EntityManager:
    # entities: List[Entity] = field(default_factory=list)
    entities: Dict[int, Entity] = field(default_factory=dict)

    def __init__(self, entity_list: List[Entity] = []):
        if entity_list is None:
            self.entities = {}
        else:
            self.entities = {entity.id: entity for entity in entity_list}

    def add(self, entity: Entity):
        if entity.id in self.entities:
            raise ValueError(f"Entity with id {entity.id} already exists.")
        self.entities[entity.id] = entity

    def remove(self, entity_id: int):
        if entity_id not in self.entities:
            raise ValueError(f"Entity with id {entity_id} does not exist.")
        del self.entities[entity_id]

    def get(self, entity_id: int) -> Entity:
        if entity_id not in self.entities:
            raise ValueError(f"Entity with id {entity_id} does not exist.")
        return self.entities[entity_id]

    def get_all(self) -> List[Entity]:
        return list(self.entities.values())

    def get_ids(self) -> List[int]:
        return list(self.entities.keys())

    def to_dict_list(self) -> List[Dict[int, Dict]]:
        return [entity.to_dict() for entity in self.entities.values()]

    def size(self) -> int:
        return len(self.entities)

    def __len__(self):
        return len(self.entities)

    def __iter__(self):
        return iter(self.entities.values())

    def plot(self, ax: plt.Axes, block_ids: List[int] = [], **kwargs) -> None:
        if block_ids:
            new_entities = [entity for entity in self.entities.values() if entity.id not in block_ids]
        else:
            new_entities = self.entities.values()

        plot_entities_on_axes(ax, new_entities, **kwargs)

    def random_one(self) -> Entity:
        if not self.entities:
            raise ValueError("No entities in the manager.")
        # return random.sample(list(self.entities.values()), 1)[0]
        # which one is better?
        return random.choice(list(self.entities.values()))

    def brief_info(self):
        return str(self.get_ids())


@dataclass(repr=True)
class HyperParams:
    """Hyper parameters for the solver.

    Attributes:
        resources_num: int = 0 资源维度数.
        map_shape: Tuple[int, int, int] = (10, 10, 10) 任务环境区域大小.
        resource_contribution_weight: float = 0.0 资源贡献权重.
        path_cost_weight: float = 0.0 路径成本权重.
        threat_loss_weight: float = 0.0 威胁权重.
        zero_resource_contribution_penalty: float = -1.0 ui 加入 tj 无资源贡献时的惩罚 (path_cost).
        max_iter: int = 0 最大迭代次数.
    """

    resources_num: int = 0  # 资源维度数
    # 任务环境区域大小
    map_shape: List[int] = field(default_factory=lambda: [10, 10, 10])
    resource_contribution_weight: float = 5.0  # 资源贡献权重
    path_cost_weight: float = 20.0  # 路径成本权重
    threat_loss_weight: float = 1.0  # 威胁权重
    zero_resource_contribution_penalty: float = -1.0  # ui 加入 tj 无资源贡献时的惩罚 (path_cost)
    max_iter: int = 10  # 最大迭代次数

    def to_dict(self) -> Dict:
        return self.__dict__

    def to_flattened_dict(self) -> Dict:
        # return {f"hyper_params.{k}": v for k, v in self.__dict__.items()}
        flattened_dict = {
            "resources_num": self.resources_num,
            "map_shape_x": self.map_shape[0],
            "map_shape_y": self.map_shape[1],
            "map_shape_z": self.map_shape[2],
            "resource_contribution_weight": self.resource_contribution_weight,
            "path_cost_weight": self.path_cost_weight,
            "threat_loss_weight": self.threat_loss_weight,
            "zero_resource_contribution_penalty": self.zero_resource_contribution_penalty,
            "max_iter": self.max_iter,
        }
        return flattened_dict

    @classmethod
    def from_flattened_dict(cls, data: Dict):
        return cls(
            resources_num=data["resources_num"],
            map_shape=[data["map_shape_x"], data["map_shape_y"], data["map_shape_z"]],
            resource_contribution_weight=data["resource_contribution_weight"],
            path_cost_weight=data["path_cost_weight"],
            threat_loss_weight=data["threat_loss_weight"],
            zero_resource_contribution_penalty=data["zero_resource_contribution_penalty"],
            max_iter=data["max_iter"],
        )

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


@dataclass
class GenParams:
    region_ranges: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0, 100), (0, 100), (0, 100)]
    )
    resources_num: int = field(default=3)


if __name__ == "__main__":
    pass
