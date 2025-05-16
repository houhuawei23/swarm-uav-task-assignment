from typing import List, Tuple, Dict, Type
import random
import numpy as np
from itertools import combinations
from math import factorial

from framework.base import HyperParams, LogLevel
from framework.uav import UAV, UAVManager, generate_uav_list, UAVGenParams
from framework.task import (
    Task,
    TaskManager,
    generate_task_list,
    TaskGenParams,
)
from framework.coalition_manager import CoalitionManager
from framework.mrta_solver import MRTASolver
import framework.utils as utils

from .utils import MRTA_CFG_Model


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

    @classmethod
    def type_name(cls):
        return "Centralized"

    def init_allocate(self):
        task_ids = self.task_manager.get_ids()
        for uav in self.uav_manager.get_all():
            task_id = random.choice(task_ids)
            self.coalition_manager.assign(uav.id, task_id)

    def allocate_once(self, uav_list: List[UAV], prefer: str = "selfish"):
        """
        Complexity: O(uav_list.size() x m x n)
        """
        # print(f"allocate_once {len(uav_list)}")
        changed = False
        # random sample allocate
        task_ids = self.task_manager.get_ids().copy()
        task_ids.append(TaskManager.free_uav_task_id)

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
                    uav,
                    taski,
                    taskj,
                    self.uav_manager,
                    self.task_manager,
                    self.coalition_manager,
                    self.hyper_params.resources_num,
                ):
                    # if true, uav leave taski, join taskj
                    # self.coalition_manager.format_print()  # check

                    self.coalition_manager.unassign(uav.id)
                    self.coalition_manager.assign(uav.id, taskj_id)
                    changed = True
                    break  # if uav changed task, break, next uav

        return changed

    def run_allocate(self):
        self.init_allocate()
        # self.coalition_manager.format_print()
        iter_cnt = 0

        not_changed_iter_cnt = 0
        sample_rate = 1 / 3
        rec_sample_size = int(max(1, self.uav_manager.size() * sample_rate))
        rec_max_iter = int(1 / sample_rate) + 1  # 期望来看，每个uav都会被抽样到
        # print(
        #     f"uav size: {self.uav_manager.size()}, rec sample size {rec_sample_size}, rec max iter {rec_max_iter}"
        # )
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
