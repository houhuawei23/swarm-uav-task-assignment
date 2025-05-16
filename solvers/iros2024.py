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


def min_max_norm(value, min_value, max_value):
    """"""
    if max_value == min_value:
        # max_value == min_value == value, return 0
        return 0
        # raise ValueError("max_value should not be equal to min_value")
    return (value - min_value) / (max_value - min_value)


def cal_uav_utility_for_task(
    uav: UAV,
    task: Task,
    uav_manager: UAVManager,
    task_manager: TaskManager,
    coalition: List[int],
    hyper_params: HyperParams,
):
    # fly cost
    fly_energy = uav.cal_fly_energy(task.position)
    max_fly_energy = max(uav.cal_fly_energy(t.position) for t in task_manager.get_all())
    min_fly_energy = min(uav.cal_fly_energy(t.position) for t in task_manager.get_all())
    # print(f"max_fly_energy: {max_fly_energy}, min_fly_energy: {min_fly_energy}")
    norm_fly_energy_cost = min_max_norm(fly_energy, min_fly_energy, max_fly_energy)

    # hover cost
    hover_energy = uav.cal_hover_energy(task.execution_time)
    max_hover_energy = max(uav.cal_hover_energy(t.execution_time) for t in task_manager.get_all())
    min_hover_energy = min(uav.cal_hover_energy(t.execution_time) for t in task_manager.get_all())
    norm_hover_energy_cost = min_max_norm(hover_energy, min_hover_energy, max_hover_energy)

    # cooperation cost
    cooperation_cost = len(coalition) ** 2 / uav_manager.size()

    # cost >= 0
    cost = norm_fly_energy_cost + norm_hover_energy_cost + cooperation_cost
    if cost == 0:
        raise ValueError("cost should not be 0")
    # task satisfaction rate = task satisfied resources num / task total resources num
    obtained_resources = calculate_obtained_resources(
        coalition, uav_manager, hyper_params.resources_num
    )
    # task.required_resources
    task_satisfaction_rate = (
        np.sum(obtained_resources >= task.required_resources) / hyper_params.resources_num
    )
    task_satisfaction_rate = min(task_satisfaction_rate, 1)
    # resource_waste_rate = used / total_input
    if np.sum(obtained_resources) == 0:
        raise ValueError("obtained_resources should not be 0")
    else:
        task_resource_waste_rate = np.sum(
            np.maximum(obtained_resources - task.required_resources, 0)
        ) / np.sum(obtained_resources)
    task_resource_use_rate = 1 - task_resource_waste_rate

    return (task_satisfaction_rate + task_resource_use_rate) / cost


def cal_uav_utility_in_colition(
    uav: UAV,
    task: Task,
    coalition: List[int],
    uav_manager: UAVManager,
    task_manager: TaskManager,
    hyper_params: HyperParams,
    debug=False,
) -> float:
    coalition_copy = coalition.copy()
    if uav.id in coalition:  # uav already in coalition
        coalition_copy.remove(uav.id)

    u_not_have = sum(
        cal_uav_utility_for_task(
            uav_manager.get(uavid),
            task,
            uav_manager,
            task_manager,
            coalition_copy,
            hyper_params,
        )
        for uavid in coalition_copy
    )

    coalition_copy.append(uav.id)

    # print(f"coalition_copy: {coalition_copy}")
    u_have = sum(
        cal_uav_utility_for_task(
            uav_manager.get(uavid),
            task,
            uav_manager,
            task_manager,
            coalition_copy,
            hyper_params,
        )
        for uavid in coalition_copy
    )
    # may be negative,
    # because if uavi has no resource contribution to task, if have:
    # task_satisfaction_rate dont change, but
    # task_resource_use_rate will be smaller
    utility = u_have - u_not_have
    return utility


# def cal_sample_size(size, min_sample_size=3):
#     rec_sample_size = max(1, size // 2)
#     sample_size = max(min_sample_size, rec_sample_size)
#     return sample_size
from . import csci2024


class IROS2024_CoalitionFormationGame(MRTASolver):
    """
    ```cpp
    Partition Alg1(R /* UAV set*/, T /* Tasks set */) {
    // Each UAV that discovers a task becomes the leader
    // UAV, while the remaining UAVs are designated as
    // follower UAVs. Let follower UAVs randomly select
    // a task tj, for_all tj in T to form an initial partition;
    S = Partition(); // init partition
    while (l < l_max) do {
        for (each ri in R, each tj in T) do {
        // According to Equation (10), each leader UAV
        // calculate the four utility values u1, u2, u3, u4
        // when UAV ri leaves its current coalition Stj
        // and joins the new coalition S′tj;
        if (u1 + u2 < u3 + u4) {
            // UAV ri leaves its current coalition Stj
            // and joins the new coalition S′tj;
        }
        }
        if (the coalition structure S has not changed) {
        l = l + 1;
        } else { // S changed
        l = 0;
        }
    }
    return S;
    }
    ```
    """

    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
    ):
        super().__init__(uav_manager, task_manager, coalition_manager, hyper_params)
        # uav_id -> int
        self.update_steps = {uav.id: 1 for uav in uav_manager.get_all()}

    @classmethod
    def type_name(cls):
        return "IROS2024_LiwangZhang"

    def allocate_once(self, uav_list: List[UAV]):
        """
        Complexity: O(uav_list.size() x m x n)

        """
        changed = False
        # random sample allocate
        task_ids = self.task_manager.get_ids().copy()
        task_ids.append(TaskManager.free_uav_task_id)
        for uav in uav_list:
            for taskj_id in task_ids:  # try to divert to another task (not have None task)
                taski_id = self.coalition_manager.get_taskid(uav.id)
                if taski_id == taskj_id:  # taski == taskj, jump (may be both None)
                    continue

                if taski_id == TaskManager.free_uav_task_id:  # not assigned to any task
                    # try to divert to taskj
                    ui = 0
                else:
                    # else, taskj is other task
                    taski = self.task_manager.get(taski_id)
                    taski_coalition_copy = self.coalition_manager.get_coalition(taski_id).copy()
                    # cal utility in taski coalition
                    # ui = cal_uav_utility_in_colition(
                    #     uav,
                    #     taski,
                    #     taski_coalition_copy,
                    #     self.uav_manager,
                    #     self.task_manager,
                    #     self.hyper_params,
                    # )
                    ui = csci2024.cal_uav_utility_in_colition(
                        uav,
                        taski,
                        taski_coalition_copy,
                        self.uav_manager,
                        self.hyper_params,
                    )  # O(n)

                if taskj_id == TaskManager.free_uav_task_id:  # try to divert to taskj (None task)
                    uj = 0
                else:
                    taskj = self.task_manager.get(taskj_id)
                    taskj_coalition_copy = self.coalition_manager.get_coalition(taskj_id).copy()
                    # uj = cal_uav_utility_in_colition(
                    #     uav,
                    #     taskj,
                    #     taskj_coalition_copy,
                    #     self.uav_manager,
                    #     self.task_manager,
                    #     self.hyper_params,
                    # )
                    uj = csci2024.cal_uav_utility_in_colition(
                        uav,
                        taskj,
                        taskj_coalition_copy,
                        self.uav_manager,
                        self.hyper_params,
                    )

                if ui < uj:
                    # uav leave taski, join taskj
                    self.coalition_manager.unassign(uav.id)
                    self.coalition_manager.assign(uav.id, taskj_id)
                    # update self.update_steps
                    self.update_steps[uav.id] += 1
                    changed = True

        return changed

    def run_allocate(self):
        """
        min O(n x m x n)
        paper: max_iter x O(n m^2) ?? -> correct: max_iter x O(n^2 m)
        有随机性 确切的时间复杂度如何估计？
        """
        # first allocate
        # each uav randomly choose a task, may be repeated
        task_ids = self.task_manager.get_ids()
        for uav in self.uav_manager.get_all():
            task_id = random.choice(task_ids)
            self.coalition_manager.assign(uav.id, task_id)

        not_changed_iter_cnt = 0
        # default_sample_size = 3
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
                print(f"iter {not_changed_iter_cnt}")
            
            if (
                not_changed_iter_cnt > self.hyper_params.max_iter
                or not_changed_iter_cnt > rec_max_iter
            ):
                if log_level >= LogLevel.INFO:
                    print(f"reach max iter {self.hyper_params.max_iter}")
                break
            # each iter randomly sample some uavs,
            # check whether they are stable (based on game theory stability)
            # Warning: if not random sample, may be deadlock!!! vibrate!!!
            sampled_uavs = random.sample(uav_list, rec_sample_size)
            
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


class IROS2024_CoalitionFormationGame_2(IROS2024_CoalitionFormationGame):
    @classmethod
    def type_name(cls):
        return "IROS2024_LiwangZhang2"

    def run_allocate(self):
        """
        min O(n x m x n)
        paper: max_iter x O(n m^2) ?? -> correct: max_iter x O(n^2 m)
        有随机性 确切的时间复杂度如何估计？
        """
        # first allocate
        # each uav randomly choose a task, may be repeated
        task_ids = self.task_manager.get_ids()
        for uav in self.uav_manager.get_all():
            task_id = random.choice(task_ids)
            self.coalition_manager.assign(uav.id, task_id)

        not_changed_iter_cnt = 0
        # default_sample_size = 3
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
                print(f"iter {not_changed_iter_cnt}")
            if (
                not_changed_iter_cnt > self.hyper_params.max_iter
                # or not_changed_iter_cnt > rec_max_iter
            ):
                if log_level >= LogLevel.INFO:
                    print(f"reach max iter {self.hyper_params.max_iter}")
                break
            # 以历史更新的次数为权重，从 uav_list 中取样
            weights = np.array([self.update_steps[uav.id] for uav in uav_list])
            sampled_uavs = random.choices(uav_list, weights=weights, k=rec_sample_size)
            changed = self.allocate_once(sampled_uavs)  # sample_size x m x n
            if not changed:
                not_changed_iter_cnt += 1
                if log_level >= LogLevel.INFO:
                    print("unchanged")
            else:
                not_changed_iter_cnt = 0
                if log_level >= LogLevel.INFO:
                    print("changed")
