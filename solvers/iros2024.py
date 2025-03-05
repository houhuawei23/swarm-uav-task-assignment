from typing import List, Tuple, Dict
from dataclasses import dataclass, field
import random
import numpy as np


from framework import *
from framework.utils import calculate_obtained_resources


def min_max_norm(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


class IROS2024_CoalitionFormationGame(CoalitionFormationGame):
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

    @staticmethod
    def cal_uav_utility_for_task(
        uav: UAV,
        task: Task,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition: List[int],
        hyper_params: HyperParams,
    ):
        # max_distance = max(uav.position.distance_to(t.position) for t in task_manager.get_all())
        # min_distance = min(uav.position.distance_to(t.position) for t in task_manager.get_all())
        # norm_distance = min_max_norm(uav.position.distance_to(task.position), min_distance, max_distance)

        # fly cost
        fly_energy = uav.cal_fly_energy(task.position)
        max_fly_energy = max(uav.cal_fly_energy(t.position) for t in task_manager.get_all())
        min_fly_energy = min(uav.cal_fly_energy(t.position) for t in task_manager.get_all())
        norm_fly_energy_cost = min_max_norm(fly_energy, min_fly_energy, max_fly_energy)

        # hover cost
        hover_energy = uav.cal_hover_energy(task.execution_time)
        max_hover_energy = max(
            uav.cal_hover_energy(t.execution_time) for t in task_manager.get_all()
        )
        min_hover_energy = min(
            uav.cal_hover_energy(t.execution_time) for t in task_manager.get_all()
        )
        norm_hover_energy_cost = min_max_norm(hover_energy, min_hover_energy, max_hover_energy)

        # cooperation cost
        cooperation_cost = len(coalition) ** 2 / uav_manager.size()

        cost = norm_fly_energy_cost + norm_hover_energy_cost + cooperation_cost

        # task satisfaction rate = task satisfied resources num / task total resources num
        obtained_resources = calculate_obtained_resources(
            coalition, uav_manager, hyper_params.resources_num
        )
        # task.required_resources
        task_satisfaction_rate = (
            np.sum(obtained_resources > task.required_resources) / hyper_params.resources_num
        )
        task_resource_waste_rate = (
            np.sum(np.maximum(obtained_resources - task.required_resources, 0))
            / hyper_params.resources_num
        )
        task_resource_use_rate = 1 - task_resource_waste_rate

        return (task_satisfaction_rate + task_resource_use_rate) / cost

    def cal_uav_utility_in_colition(
        self, uav: UAV, task: Task, coalition: List[int], debug=False
    ) -> float:
        coalition_copy = coalition.copy()
        if uav.id in coalition:  # uav already in coalition
            coalition_copy.remove(uav.id)

        u_not_have = sum(
            self.cal_uav_utility_for_task(
                uav, task, self.uav_manager, self.task_manager, coalition_copy, self.hyper_params
            )
            for uav in self.uav_manager.get_all()
        )

        coalition_copy.append(uav.id)

        u_have = sum(
            self.cal_uav_utility_for_task(
                uav, task, self.uav_manager, self.task_manager, coalition_copy, self.hyper_params
            )
            for uav in self.uav_manager.get_all()
        )
        utility = u_have - u_not_have
        return utility

    def allocate_once(self, uav_list: List[UAV], debug=False):
        changed = False
        if len(uav_list) == self.uav_manager.size():
            # first allocate
            # each uav randomly choose a task, may be repeated
            task_ids = self.task_manager.get_ids()
            for uav in uav_list:
                task_id = random.choice(task_ids)
                self.coalition_manager.assign(uav, self.task_manager.get(task_id))
        else:  # random sample allocate
            task_list = self.task_manager.get_all()
            for uav in uav_list:
                for taskj in task_list:
                    taski_id = self.coalition_manager.get_taskid_by_uavid(uav.id)
                    if taski_id == taskj.id:
                        continue
                    # else, taskj is other task
                    taski = self.task_manager.get(taski_id)
                    taski_coalition_copy = self.coalition_manager.get_coalition(taski_id).copy()
                    # cal utility in taski coalition
                    ui = self.cal_uav_utility_in_colition(
                        uav, taski, coalition=taski_coalition_copy, debug=debug
                    )
                    # cal utility in taskj coalition
                    taskj_coalition_copy = self.coalition_manager.get_coalition(taskj.id).copy()
                    uj = self.cal_uav_utility_in_colition(
                        uav, taskj, coalition=taskj_coalition_copy, debug=debug
                    )
                    if ui < uj:
                        # uav leave taski, join taskj
                        self.coalition_manager.unassign(uav)
                        self.coalition_manager.assign(uav, taskj)
                        changed = True
        return changed

    def run_allocate(self, debug=False):
        self.allocate_once(self.uav_manager.get_all(), debug=debug)

        iter_cnt = 0
        sample_size = 3
        while True:
            print(f"iter {iter_cnt}")
            if iter_cnt > self.hyper_params.max_iter:
                print(f"reach max iter {self.hyper_params.max_iter}")
                break
            # each iter randomly sample some uavs,
            # check whether they are stable (based on game theory stability)
            sampled_uavs = random.sample(self.uav_manager.get_all(), sample_size)
            changed = self.allocate_once(sampled_uavs, debug=debug)
            if not changed:
                iter_cnt += 1
            else:
                iter_cnt = 0
