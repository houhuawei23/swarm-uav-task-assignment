from typing import List, Tuple, Dict, Type
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

from framework.base import HyperParams, LogLevel
from framework.uav import UAV, UAVManager, generate_uav_list, UAVGenParams
from framework.task import (
    TaskManager,
    generate_task_list,
    TaskGenParams,
)
from framework.coalition_manager import CoalitionManager
from framework.mrta_solver import MRTASolver
import framework.utils as utils

from .utils import MRTA_CFG_Model, MRTA_CFG_Model_HyperParams


log_level: LogLevel = LogLevel.SILENCE


class CentralizedSolver(MRTASolver):
    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
    ):
        super().__init__(uav_manager, task_manager, coalition_manager, hyper_params)
        map_shape: List[float] = utils.calculate_map_shape_on_mana(uav_manager, task_manager)
        max_distance = max(map_shape)
        max_uav_value = max(uav.value for uav in uav_manager.get_all())
        self.model_hparams = MRTA_CFG_Model_HyperParams(
            max_distance=max_distance,
            max_uav_value=max_uav_value,
            # w_sat=10.0,
            # w_waste=1,
            # w_dist=1,
            # w_threat=1,
            w_sat=70.0,
            w_waste=1,
            w_dist=12,
            w_threat=1,
        )

    @classmethod
    def type_name(cls):
        return "Centralized"

    def init_allocate(self):
        task_ids = self.task_manager.get_ids()
        for uav in self.uav_manager.get_all():
            task_id = random.choice(task_ids)
            self.coalition_manager.assign(uav.id, task_id)

    def init_allocate_beta(self):
        """
        每一轮迭代中，给每个任务分配一个未经初始化分配的无人机。
        首先计算所有未经初始化分配的无人机加入各个任务联盟的边际收益，形成无人机-任务收益矩阵。
        由于每个任务只能分配一个无人机，采用最大加权匹配算法，使本轮迭代收益最大化。
        如此迭代，直至所有无人机都分配到任务，形成初始联盟结构。

        Returns:
            bool: True if allocation was successful
        """
        # 获取所有任务ID（不包括free_uav_task）
        task_ids = [
            tid for tid in self.task_manager.get_ids() if tid != TaskManager.free_uav_task_id
        ]

        # 初始时所有UAV都是未分配的
        unassigned_uavs = self.uav_manager.get_all()

        # 当还有未分配的UAV时，继续迭代
        while unassigned_uavs:
            # 构建收益矩阵
            benefit_matrix = np.zeros((len(unassigned_uavs), len(task_ids)))

            # 计算每个未分配UAV对每个任务的边际收益
            for i, uav in enumerate(unassigned_uavs):
                for j, task_id in enumerate(task_ids):
                    task = self.task_manager.get(task_id)
                    # 获取当前任务的联盟
                    current_coalition = self.coalition_manager.get_coalition(task_id)
                    current_coalition_uavs = [
                        self.uav_manager.get(uid) for uid in current_coalition
                    ]

                    # 计算当前联盟的效用
                    before = MRTA_CFG_Model.cal_coalition_eval(
                        task,
                        current_coalition_uavs,
                        self.hyper_params.resources_num,
                        self.model_hparams,
                    )

                    # 计算加入新UAV后的效用
                    new_coalition_uavs = current_coalition_uavs + [uav]
                    after = MRTA_CFG_Model.cal_coalition_eval(
                        task,
                        new_coalition_uavs,
                        self.hyper_params.resources_num,
                        self.model_hparams,
                    )

                    # 边际收益
                    benefit_matrix[i, j] = after - before

            # 使用匈牙利算法找到最优匹配
            row_indices, col_indices = linear_sum_assignment(benefit_matrix, maximize=True)

            # 应用分配结果
            removed_uavs = []
            for row_idx, col_idx in zip(row_indices, col_indices):
                uav = unassigned_uavs[row_idx]
                task_id = task_ids[col_idx]
                # 只分配正收益的匹配
                if benefit_matrix[row_idx, col_idx] > 0:
                    self.coalition_manager.assign(uav.id, task_id)
                else:
                    # 如果收益为负，分配到free_uav_task
                    self.coalition_manager.assign(uav.id, TaskManager.free_uav_task_id)

                removed_uavs.append(uav)

            unassigned_uavs = [uav for uav in unassigned_uavs if uav not in removed_uavs]

        return True

    def allocate_once(self, uav_list: List[UAV], prefer: str = "selfish"):
        """
        Complexity: O(uav_list.size() x m x n)
        """
        # print(f"allocate_once {len(uav_list)}")
        changed = False
        # random sample allocate
        task_ids = self.task_manager.get_ids().copy()

        # change traverse
        for uav in uav_list:
            # print(f"uav {uav.id}")
            for taskj_id in task_ids:  # try to divert to another task (not have None task)
                taski_id = self.coalition_manager.get_taskid(uav.id)
                if taski_id == taskj_id:  # taski == taskj, jump (may be both None)
                    continue
                taski = self.task_manager.get(taski_id)
                taskj = self.task_manager.get(taskj_id)
                prefer = "cooperative"
                prefer_func = MRTA_CFG_Model.get_prefer_func(prefer)
                if prefer_func(
                    uav=uav,
                    task_p=taski,
                    task_q=taskj,
                    uav_manager=self.uav_manager,
                    task_manager=self.task_manager,
                    coalition_manager=self.coalition_manager,
                    resources_num=self.hyper_params.resources_num,
                    model_hparams=self.model_hparams,
                ):
                    # if true, uav leave taski, join taskj
                    # print(f"uav {uav.id} leave t{taski_id}, join t{taskj_id}")
                    self.coalition_manager.format_print()  # check

                    self.coalition_manager.unassign(uav.id)
                    self.coalition_manager.assign(uav.id, taskj_id)
                    changed = True
                    break  # if uav changed task, break, next uav

        return changed

    def run_allocate(self):
        # 使用新的初始化方法
        self.init_allocate_beta()

        # 后续优化过程保持不变
        not_changed_iter_cnt = 0
        sample_rate = 1 / 3
        rec_sample_size = int(max(1, self.uav_manager.size() * sample_rate))
        rec_max_iter = int(1 / sample_rate) + 10

        uav_list = self.uav_manager.get_all()
        # Complexity: max_iter x O(n x m x n)
        while True:  # max_iter or 1/sample_rate
            if log_level >= LogLevel.INFO:
                print(f"not_changed_iter_cnt {not_changed_iter_cnt}")

            if (
                not_changed_iter_cnt > self.hyper_params.max_iter
                or not_changed_iter_cnt > rec_max_iter
            ):
                if log_level >= LogLevel.INFO:
                    print(f"reach max iter {self.hyper_params.max_iter}")
                break
            sampled_uavs = random.sample(uav_list, rec_sample_size)
            # 以历史更新的次数为权重，从 uav_list 中取样
            changed = self.allocate_once(sampled_uavs)  # sample_size x m x n
            # changed = self.allocate_once(self.uav_manager.get_all(), debug=debug)
            if not changed:
                not_changed_iter_cnt += 1
                if log_level >= LogLevel.INFO:
                    print("unchanged")
            else:
                not_changed_iter_cnt = 0
                if log_level >= LogLevel.INFO:
                    print("changed")

        # print
        # self.coalition_manager.format_print()


def test_divert_t0():
    resources_num = 5
    uav_gen = UAVGenParams(resources_num=resources_num)
    task_gen = TaskGenParams(resources_num=resources_num)

    task_list = generate_task_list(5, task_gen)
    uav_list = generate_uav_list(10, uav_gen)
    hyper_params = HyperParams(
        resources_num=resources_num,
        map_shape=utils.calculate_map_shape_on_list(uav_list, task_list),
    )
    task_manager = TaskManager(task_list, resources_num)
    uav_manager = UAVManager(uav_list)
    # print(task_manager.brief_info())
    # print(uav_manager.brief_info())
    task_manager.format_print()
    uav_manager.format_print()

    t = task_list[0]

    coalition_manager = CoalitionManager(uav_manager.get_ids(), task_manager.get_ids())
    for uav in uav_list:
        coalition_manager.assign(uav.id, t.id)

    tp = t
    tq = task_manager.get_free_uav_task(resources_num)

    u = uav_list[0]

    is_selfish = MRTA_CFG_Model.selfish_prefer(
        u, tp, tq, uav_manager, task_manager, coalition_manager, resources_num
    )
    print("is_selfish", is_selfish)

    is_pareto = MRTA_CFG_Model.pareto_prefer(
        u, tp, tq, uav_manager, task_manager, coalition_manager, resources_num
    )
    print("is_pareto", is_pareto)

    is_cooperative = MRTA_CFG_Model.cooperative_prefer(
        u, tp, tq, uav_manager, task_manager, coalition_manager, resources_num
    )
    print("is_cooperative", is_cooperative)


if __name__ == "__main__":
    test_divert_t0()
