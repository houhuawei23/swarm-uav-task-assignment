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
            w_sat=150.0,
            w_waste=1,
            w_dist=15,
            w_threat=1,
            # w_sat=70.0,
            # w_waste=1,
            # w_dist=12,
            # w_threat=1,
        )
        # self.model_hparams = MRTA_CFG_Model_HyperParams(
        #     max_distance=max_distance,
        #     max_uav_value=max_uav_value,
        #     # w_sat=10.0,
        #     # w_waste=1,
        #     # w_dist=1,
        #     # w_threat=1,
        #     w_sat=hyper_params.resource_contribution_weight,
        #     w_waste=hyper_params.resource_waste_weight,
        #     w_dist=hyper_params.path_cost_weight,
        #     w_threat=hyper_params.threat_loss_weight,
        # )

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

    def try_divert(self, uav_list: List[UAV], prefer: str = "cooperative"):
        # random sample allocate
        task_ids = self.task_manager.get_ids().copy()

        changed = False
        # traverse: try to divert
        for uav in uav_list:
            # print(f"uav {uav.id}")
            for taskj_id in task_ids:  # try to divert to another task (not have None task)
                taski_id = self.coalition_manager.get_taskid(uav.id)
                if taski_id == taskj_id:  # taski == taskj, jump (may be both None)
                    continue
                taski = self.task_manager.get(taski_id)
                taskj = self.task_manager.get(taskj_id)
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
                    # self.coalition_manager.format_print()  # check

                    self.coalition_manager.unassign(uav.id)
                    self.coalition_manager.assign(uav.id, taskj_id)
                    changed = True
                    # break  # if uav changed task, break, next uav
        return changed

    def try_exchange(self, uav_list: List[UAV], prefer: str = "cooperative"):
        changed = False

        # traverse: try to exchange, between uavi and uavj
        # TODO:
        for uavi_idx in range(len(uav_list)):
            for uavj_idx in range(uavi_idx + 1, len(uav_list)):
                uavi = uav_list[uavi_idx]
                uavj = uav_list[uavj_idx]
                if uavi.id == uavj.id:
                    continue
                taski_id = self.coalition_manager.get_taskid(uavi.id)
                taskj_id = self.coalition_manager.get_taskid(uavj.id)
                if taski_id == taskj_id:
                    continue
                # taski = self.task_manager.get(taski_id)
                # taskj = self.task_manager.get(taskj_id)
                if MRTA_CFG_Model.cooperative_exchange_prefer(
                    uavi,
                    uavj,
                    self.uav_manager,
                    self.task_manager,
                    self.coalition_manager,
                    self.hyper_params.resources_num,
                    self.model_hparams,
                ):
                    self.coalition_manager.unassign(uavi.id)
                    self.coalition_manager.assign(uavi.id, taskj_id)
                    self.coalition_manager.unassign(uavj.id)
                    self.coalition_manager.assign(uavj.id, taski_id)
                    changed = True
                    # break  # if uav changed task, break, next uav
        return changed

    def allocate_once(
        self, uav_list: List[UAV], prefer: str = "cooperative", try_exchange: bool = False
    ):
        """
        Complexity: O(uav_list.size() x m x n)
        """
        # print(f"allocate_once {len(uav_list)}")
        changed = False
        changed |= self.try_divert(uav_list, prefer)
        if try_exchange:
            changed |= self.try_exchange(uav_list, prefer)
        return changed

    def run_allocate(self, init_method: str = "beta"):
        self._run_allocate(init_method=init_method, prefer="cooperative", try_exchange=False)

    def _run_allocate(
        self,
        init_method: str = "beta",
        prefer: str = "cooperative",
        try_exchange: bool = False,
    ):
        """Run the allocation algorithm with specified initialization method and preferences.

        Args:
            init_method: Method to use for initial allocation ("beta", "random", or "none")
            prefer: Preference strategy for allocation ("cooperative", "selfish", or "pareto")
            try_exchange: Whether to attempt UAV exchanges during allocation
        """
        # Initialize allocation based on specified method
        if init_method == "beta":
            self.init_allocate_beta()
        elif init_method == "random":
            self.init_allocate()
        elif init_method == "none":
            pass
        else:
            raise ValueError(f"Invalid init method: {init_method}")

        # Setup iteration parameters
        max_not_changed_iter = 5
        not_changed_iter_cnt = 0
        iter = 0
        sample_rate = 1 / 3
        rec_sample_size = int(max(1, self.uav_manager.size() * sample_rate))
        rec_max_not_changed_iter = int(1 / sample_rate) + 10

        uav_list = self.uav_manager.get_all()

        def log_iteration_status():
            """Helper function to log iteration status"""
            if log_level >= LogLevel.INFO:
                print(f"Iteration {iter}: {not_changed_iter_cnt} unchanged iterations")

        def log_termination_reason(reason: str):
            """Helper function to log termination reason"""
            if log_level >= LogLevel.INFO:
                print(f"Terminating: {reason}")

        # Main allocation loop
        while True:
            log_iteration_status()

            # Check termination conditions
            if not_changed_iter_cnt > max_not_changed_iter:
                log_termination_reason(f"Reached max unchanged iterations ({max_not_changed_iter})")
                break
            if not_changed_iter_cnt > rec_max_not_changed_iter:
                log_termination_reason(
                    f"Reached recommended max unchanged iterations ({rec_max_not_changed_iter})"
                )
                break
            if iter > self.hyper_params.max_iter:
                log_termination_reason(f"Reached max iterations ({self.hyper_params.max_iter})")
                break

            # Sample UAVs and attempt allocation
            sampled_uavs = random.sample(uav_list, rec_sample_size)
            changed = self.allocate_once(sampled_uavs, prefer=prefer, try_exchange=try_exchange)

            # Update iteration counters
            if not changed:
                not_changed_iter_cnt += 1
                if log_level >= LogLevel.INFO:
                    print("No changes in this iteration")
            else:
                not_changed_iter_cnt = 0
                if log_level >= LogLevel.INFO:
                    print("Changes detected in this iteration")

            iter += 1

        # print
        # self.coalition_manager.format_print()


class CentralizedSolver_Exchange(CentralizedSolver):
    @classmethod
    def type_name(cls):
        return "Centralized_Exchange"

    def run_allocate(self):
        self._run_allocate(init_method="beta", prefer="cooperative", try_exchange=True)


class CentralizedSolver_Selfish(CentralizedSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefer = "selfish"

    @classmethod
    def type_name(cls):
        return "Centralized_Selfish"

    def run_allocate(self):
        self._run_allocate(init_method="beta", prefer=self.prefer)


class CentralizedSolver_Pareto(CentralizedSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefer = "pareto"

    @classmethod
    def type_name(cls):
        return "Centralized_Pareto"

    def run_allocate(self):
        self._run_allocate(init_method="beta", prefer=self.prefer)


class CentralizedSolver_RandomInit(CentralizedSolver):
    @classmethod
    def type_name(cls):
        return "Centralized_RandomInit"

    def run_allocate(self):
        self._run_allocate(init_method="random")


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
