from typing import List, Tuple, Dict
from dataclasses import dataclass, field
import random
import numpy as np

from framework.base import HyperParams, LogLevel
from framework.uav import UAV, UAVManager
from framework.task import Task, TaskManager
from framework.coalition_manager import CoalitionManager
from framework.mrta_solver import MRTASolver
from framework.utils import calculate_obtained_resources

log_level: LogLevel = LogLevel.SILENCE


from . import iros2024


class AcutionBiddingSolverKimi(MRTASolver):
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
    ):
        super().__init__(uav_manager, task_manager, coalition_manager, hyper_params)

    def run_allocate(self):
        """
        拍卖竞标算法进行任务分配
        """
        # 初始化每个任务的当前出价和中标无人机
        task_bids = {task.id: {"bid": 0, "winner": None} for task in self.task_manager.get_all()}

        # 迭代次数
        iter_cnt = 0
        while True:
            # print(f"iter {iter_cnt}")
            if iter_cnt > self.hyper_params.max_iter:
                # print(f"reach max iter {self.hyper_params.max_iter}")
                break
            iter_cnt += 1

            # 遍历每个无人机，进行出价
            for uav in self.uav_manager.get_all():
                # 计算该无人机对每个任务的估价
                task_utilities = {}
                for task in self.task_manager.get_all():
                    utility = self.calculate_uav_task_utility(uav, task)
                    task_utilities[task.id] = utility

                # 选择估价最高的任务进行出价
                target_task_id = max(task_utilities, key=task_utilities.get)
                target_task_bid = task_utilities[target_task_id]

                # 如果出价高于当前最高出价，则更新中标无人机
                if target_task_bid > task_bids[target_task_id]["bid"]:
                    task_bids[target_task_id]["bid"] = target_task_bid
                    task_bids[target_task_id]["winner"] = uav.id

            # 检查是否达到稳定状态
            if self.check_stability(task_bids):
                break

        # 根据中标结果分配任务
        for task_id, bid_info in task_bids.items():
            if bid_info["winner"] is not None:
                self.coalition_manager.assign(bid_info["winner"], task_id)

        return task_bids

    def calculate_uav_task_utility(self, uav: UAV, task: Task) -> float:
        """
        计算无人机对任务的估价
        """
        # 飞行成本
        fly_energy = uav.cal_fly_energy(task.position)
        max_fly_energy = max(uav.cal_fly_energy(t.position) for t in self.task_manager.get_all())
        min_fly_energy = min(uav.cal_fly_energy(t.position) for t in self.task_manager.get_all())
        norm_fly_energy_cost = iros2024.min_max_norm(fly_energy, min_fly_energy, max_fly_energy)

        # 悬停成本
        hover_energy = uav.cal_hover_energy(task.execution_time)
        max_hover_energy = max(
            uav.cal_hover_energy(t.execution_time) for t in self.task_manager.get_all()
        )
        min_hover_energy = min(
            uav.cal_hover_energy(t.execution_time) for t in self.task_manager.get_all()
        )
        norm_hover_energy_cost = iros2024.min_max_norm(
            hover_energy, min_hover_energy, max_hover_energy
        )

        # 合作成本
        cooperation_cost = (
            len(self.coalition_manager.get_coalition(task.id)) ** 2 / self.uav_manager.size()
        )

        # 成本
        cost = norm_fly_energy_cost + norm_hover_energy_cost + cooperation_cost

        # 任务满足率
        obtained_resources = calculate_obtained_resources(
            [uav.id], self.uav_manager, self.hyper_params.resources_num
        )
        task_satisfaction_rate = (
            np.sum(obtained_resources >= task.required_resources) / self.hyper_params.resources_num
        )
        task_satisfaction_rate = min(task_satisfaction_rate, 1)

        # 资源利用率
        if np.sum(obtained_resources) == 0:
            raise ValueError("obtained_resources should not be 0")
        else:
            task_resource_waste_rate = np.sum(
                np.maximum(obtained_resources - task.required_resources, 0)
            ) / np.sum(obtained_resources)
        task_resource_use_rate = 1 - task_resource_waste_rate

        # 估价
        utility = (task_satisfaction_rate + task_resource_use_rate) / cost
        return utility

    def check_stability(self, task_bids: Dict) -> bool:
        """
        检查是否达到稳定状态
        """
        # 如果所有任务的中标无人机没有变化，则认为达到稳定状态
        for task_id, bid_info in task_bids.items():
            if bid_info["winner"] is None:
                return False
            current_winner = bid_info["winner"]
            current_bid = bid_info["bid"]
            for uav in self.uav_manager.get_all():
                if uav.id == current_winner:
                    continue
                utility = self.calculate_uav_task_utility(uav, self.task_manager.get(task_id))
                if utility > current_bid:
                    return False
        return True
