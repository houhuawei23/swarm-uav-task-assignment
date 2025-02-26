import json
import random
from utils import format_json


# 生成 UAV 数据
def generate_uavs(
    num_uavs,
    region_range=[(0, 50), (0, 50), (0, 0)],
    resources_range=(5, 15),
    value_range=(10, 20),
    speed_range=(20, 30),
    resources_num=2,
):
    x_range, y_range, z_ranage = region_range
    uavs = []
    for i in range(1, num_uavs + 1):
        uav = {
            "id": i,
            "resources": [
                random.randint(*resources_range) for _ in range(resources_num)
            ],  # 随机生成资源
            "position": [
                random.randint(x_range[0], x_range[1]),
                random.randint(y_range[0], y_range[1]),
                random.randint(z_ranage[0], z_ranage[1]),
            ],  # 随机生成位置
            "value": random.randint(*value_range),  # 随机生成价值
            "max_speed": random.randint(*speed_range),  # 随机生成最大速度
        }
        uavs.append(uav)
    return uavs


# 生成 Task 数据
def generate_tasks(
    num_tasks,
    uavs,
    region_range=[(0, 50), (0, 50), (0, 0)],
    time_window_range=[(0, 10), (90, 100)],
    threat_range=(0.1, 0.9),
    resources_range=(50, 75),
    resources_num=2,
):
    x_range, y_range, z_ranage = region_range
    tasks = []
    # 计算任务所需的资源（确保任务资源需求不超过 UAV 资源总和）
    total_resources = [
        sum(uav["resources"][0] for uav in uavs),
        sum(uav["resources"][1] for uav in uavs),
    ]
    for i in range(1, num_tasks + 1):
        # required_resources = [
        #     random.randint(
        #         1, min(10, total_resources[0])
        #     ),  # 确保资源需求不超过 UAV 资源总和
        #     random.randint(1, min(10, total_resources[1])),
        # ]
        required_resources = [
            random.randint(*resources_range) for _ in range(resources_num)
        ]
        task = {
            "id": i,
            "required_resources": required_resources,
            "position": [
                random.randint(*x_range),
                random.randint(*y_range),
                random.randint(*z_ranage),
            ],  # 随机生成位置
            "time_window": [
                random.randint(*time_window_range[0]),
                random.randint(*time_window_range[1]),
            ],
            # "threat": random.uniform(*threat_range),  # 随机生成威胁
            "threat": round(random.uniform(*threat_range), 2),  # 随机生成威胁
        }
        tasks.append(task)
    return tasks


# 生成数据并保存为 JSON 文件
def generate_data(num_uavs, num_tasks, output_file, params):
    (
        region_range,
        resources_range,
        value_range,
        speed_range,
        time_window_range,
        threat_range,
    ) = params.values()
    uavs = generate_uavs(
        num_uavs,
        region_range=region_range,
        resources_range=resources_range,
        value_range=value_range,
        speed_range=speed_range,
    )
    tasks = generate_tasks(
        num_tasks,
        uavs,
        region_range=region_range,
        time_window_range=time_window_range,
        threat_range=threat_range,
    )
    data = {"uavs": uavs, "tasks": tasks}
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {output_file}")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coalition Formation Game Simulation")
    # test_case
    parser.add_argument(
        "--output",
        type=str,
        default="./tests/case1.json",
        help="path to the test case file",
    )
    # parse args
    args = parser.parse_args()
    out_path = args.output

    region_range = [(0, 100), (0, 100), (0, 0)]
    resources_range = (5, 25)
    resources_num = 2
    value_range = (25, 50)
    speed_range = (20, 30)

    # region_range = [(0, 50), (0, 50), (0, 0)]
    time_window_range = [(0, 10), (90, 100)]
    threat_range = (0.1, 0.9)

    params = {
        "region_range": region_range,
        "resources_range": resources_range,
        "value_range": value_range,
        "speed_range": speed_range,
        "time_window_range": time_window_range,
        "threat_range": threat_range,
    }
    generate_data(num_uavs=50, num_tasks=10, output_file=out_path, params=params)
    format_json(out_path)
