from typing import List, Dict, Any, Tuple

import json
import random
import argparse

from framework.utils import format_json

from dataclasses import dataclass, field


@dataclass
class Params:
    region_ranges: List[Tuple[float, float]]
    resources_num: int


@dataclass
class UAVParams(Params):
    resources_range: Tuple[float, float]
    value_range: Tuple[float, float]
    speed_range: Tuple[float, float]

    uav_mass_range: Tuple[float, float] = field(default=None)
    fly_energy_per_time_range: Tuple[float, float] = field(default=None)
    hover_energy_per_time_range: Tuple[float, float] = field(default=None)
    comm_bandwidth_range: Tuple[float, float] = field(default=None)
    trans_power_range: Tuple[float, float] = field(default=None)


@dataclass
class TaskParams(Params):
    required_resources_range: Tuple[float, float]
    time_window_ranges: List[Tuple[float, float]]
    threat_range: Tuple[float, float]
    execution_time_range: Tuple[float, float] = field(default=None)


precision = 2


def generate_uavs_beta(num_uavs: int, params: UAVParams):
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


# 生成 Task 数据
def generate_tasks_beta(num_tasks: int, params: TaskParams):
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


def gen_iros2024_case(out_path):
    num_uavs = 50
    num_tasks = 5

    region_ranges = [(100, 900), (100, 900), (0, 0)]  # [0, 1000]x[0, 1000] m
    resources_num = 5
    # UAV
    resources_range = (5, 10)
    value_range = (25, 50)
    speed_range = (20, 30)  # m/s
    uav_mass_range = (10, 15)  # kg
    fly_energy_per_time_range = (1, 3)  # mW
    hover_energy_per_time_range = (1, 3)  # mW
    comm_bandwidth_range = (5, 15)  # kHz
    trans_power_range = (20, 30)  # dBm

    uav_params = UAVParams(
        region_ranges=region_ranges,
        resources_num=resources_num,
        resources_range=resources_range,
        value_range=value_range,
        speed_range=speed_range,
        uav_mass_range=uav_mass_range,
        fly_energy_per_time_range=fly_energy_per_time_range,
        hover_energy_per_time_range=hover_energy_per_time_range,
        comm_bandwidth_range=comm_bandwidth_range,
        trans_power_range=trans_power_range,
    )

    # Task
    required_resources_range = (25, 45)
    time_window_ranges = [(0, 10), (90, 100)]
    threat_range = (0.1, 0.9)

    task_params = TaskParams(
        region_ranges=region_ranges,
        resources_num=resources_num,
        required_resources_range=required_resources_range,
        execution_time_range=(10, 15),
        time_window_ranges=time_window_ranges,
        threat_range=threat_range,
    )

    # Communication
    antenna_gain_const = 1  # 天线增益常数
    attenuation_factor = 3  # 衰减因子
    noise_power_range = [-60, -70]  # dBm

    uavs = generate_uavs_beta(num_uavs, uav_params)
    tasks = generate_tasks_beta(num_tasks, task_params)
    data = {
        "resources_num": resources_num,
        "num_uavs": num_uavs,
        "num_tasks": num_tasks,
        "uavs": uavs,
        "tasks": tasks,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {out_path}")
    format_json(out_path)


def gen_csci2024_data(out_path):
    num_uavs = 50
    num_tasks = 5

    region_ranges = [(0, 100), (0, 100), (0, 0)]
    resources_num = 2
    # UAV
    resources_range = (5, 25)
    value_range = (25, 50)
    speed_range = (20, 30)

    # Task
    required_resources_range = (25, 45)
    time_window_ranges = [(0, 10), (90, 100)]
    threat_range = (0.1, 0.9)

    uav_params = UAVParams(
        region_ranges=region_ranges,
        resources_num=resources_num,
        resources_range=resources_range,
        value_range=value_range,
        speed_range=speed_range,
    )

    task_params = TaskParams(
        region_ranges=region_ranges,
        resources_num=resources_num,
        required_resources_range=required_resources_range,
        time_window_ranges=time_window_ranges,
        threat_range=threat_range,
    )
    # generate_data(num_uavs=50, num_tasks=10, output_file=out_path, params=params)
    uavs = generate_uavs_beta(num_uavs, uav_params)
    tasks = generate_tasks_beta(num_tasks, task_params)
    data = {
        "resources_num": resources_num,
        "num_uavs": num_uavs,
        "num_tasks": num_tasks,
        "uavs": uavs,
        "tasks": tasks,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {out_path}")
    format_json(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coalition Formation Game Simulation")
    # test_case
    parser.add_argument(
        "--output",
        type=str,
        default="./tests/case1.json",
        help="path to the test case file",
    )
    parser.add_argument(
        "--num_uavs",
        type=int,
        default=50,
        help="number of uavs",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=5,
        help="number of tasks",
    )
    parser.add_argument(
        "--choice",
        type=str,
        default="csci2024",
        help="choice of data generation",
    )

    # parse args
    args = parser.parse_args()
    out_path = args.output
    if args.choice == "csci2024":
        print("Generating CSCI2024 data")
        gen_csci2024_data(out_path)
    elif args.choice == "iros2024":
        print("Generating IROS2024 data")
        gen_iros2024_case(out_path)

