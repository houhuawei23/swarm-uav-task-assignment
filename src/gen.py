import json
import random
from utils import format_json


# 生成 UAV 数据
def generate_uavs(num_uavs):
    uavs = []
    for i in range(1, num_uavs + 1):
        uav = {
            "id": i,
            "resources": [random.randint(1, 5), random.randint(1, 5)],  # 随机生成资源
            "position": [
                random.randint(0, 50),
                random.randint(0, 50),
                0,
            ],  # 随机生成位置
            "value": random.randint(10, 20),  # 随机生成价值
            "max_speed": random.randint(20, 30),  # 随机生成最大速度
        }
        uavs.append(uav)
    return uavs


# 生成 Task 数据
def generate_tasks(num_tasks, uavs):
    tasks = []
    for i in range(1, num_tasks + 1):
        # 计算任务所需的资源（确保任务资源需求不超过 UAV 资源总和）
        total_resources = [
            sum(uav["resources"][0] for uav in uavs),
            sum(uav["resources"][1] for uav in uavs),
        ]
        required_resources = [
            random.randint(
                1, min(10, total_resources[0])
            ),  # 确保资源需求不超过 UAV 资源总和
            random.randint(1, min(10, total_resources[1])),
        ]
        task = {
            "id": i,
            "required_resources": required_resources,
            "position": [
                random.randint(0, 50),
                random.randint(0, 50),
                0,
            ],  # 随机生成位置
            "time_window": [0, 100],  # 固定时间窗口
            "threat": round(random.uniform(0.1, 0.9), 1),  # 随机生成威胁值
        }
        tasks.append(task)
    return tasks


# 生成数据并保存为 JSON 文件
def generate_data(num_uavs, num_tasks, output_file):
    uavs = generate_uavs(num_uavs)
    tasks = generate_tasks(num_tasks, uavs)
    data = {"uavs": uavs, "tasks": tasks}
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    out_path = "../tests/case3.json"
    generate_data(num_uavs=5, num_tasks=5, output_file=out_path)
    format_json(out_path)
