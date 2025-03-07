from typing import List, Tuple, Dict
from dataclasses import dataclass, field
import random
import numpy as np


from framework import *


@dataclass
class Message:
    uav_id: int
    changed: bool
    uav_update_step_dict: Dict[int, int]
    task2coalition: Dict[int, List[int]]
    uav2task: Dict[int, int]


from .iros2024 import cal_uav_utility_in_colition


@dataclass
class AutoUAV(UAV):
    """
    r(tj, Sj^ck) = vj * log(min(wj^ck, |Sj^ck|) + e, wj^ck)

    ui(tj, Sj) = (Sum[ck in C]{ r(tj, Sj^ck) }) / |Sj^ck| - cost(i, j)

    cost(i, j) = distance(i, j)
    """

    # 知道自己的临近无人机的id uav_ids => 知道临近uav的所有信息
    # 知道全局的任务(包括任务信息)
    uav_manager: UAVManager = field(default=None, init=False)
    task_manager: TaskManager = field(default=None, init=False)
    coalition_manager: CoalitionManager = field(default=None, init=False)
    hyper_params: HyperParams = field(default=None, init=False)

    uav_update_step_dict: Dict[int, int] = field(default_factory=dict, init=False)
    changed: bool = field(default=False, init=False)

    def __post_init__(self):
        super().__post_init__()

    def init(self, uav_manager: UAVManager, task_manager: TaskManager, hyper_params: HyperParams):
        self.uav_manager = uav_manager
        self.task_manager = task_manager
        self.coalition_manager = CoalitionManager(uav_manager.get_ids(), task_manager.get_ids())
        self.hyper_params = hyper_params

        # self.uav_stable_dict = {uav_id: False for uav_id in uav_manager.get_ids()}
        self.uav_update_step_dict = {uav_id: 0 for uav_id in uav_manager.get_ids()}

    def send_msg(self) -> Message:
        """
        broadcast msg to its neighbors.
        """
        msg = Message(
            uav_id=self.id,
            changed=self.changed,
            uav_update_step_dict=self.uav_update_step_dict.copy(),
            task2coalition=self.coalition_manager.get_task2coalition().copy(),
            uav2task=self.coalition_manager.get_uav2task().copy(),
        )
        return msg

    def receive_and_update(self, msgs: List[Message]) -> bool:
        """
        receive msgs from its neighbors, update local coalition.
        """
        receive_changed = False
        for msg in msgs:
            print(f"msg: {msg}")
            for uav_id in self.uav_manager.get_ids():
                if self.uav_update_step_dict[uav_id] < msg.uav_update_step_dict[uav_id]:
                    self.coalition_manager.unassign(uav_id)
                    self.coalition_manager.assign(uav_id, msg.uav2task[uav_id])
                    self.uav_update_step_dict[uav_id] = msg.uav_update_step_dict[uav_id]
                    receive_changed = True
                    print(f"receive_changed: {receive_changed}")
        self.changed = self.changed or receive_changed
        return receive_changed

    def try_divert(self) -> bool:
        """ """
        # pass
        task_ids = self.task_manager.get_ids().copy()
        task_ids.append(None)  # special, None means no task
        divert_changed = False
        for taskj_id in task_ids:
            taski_id = self.coalition_manager.get_taskid_by_uavid(self.id)
            if taski_id == taskj_id:
                continue

            if taski_id is None:
                taski = None
                ui = 0
            else:
                taski = self.task_manager.get(taski_id)
                taski_coalition_copy = self.coalition_manager.get_coalition(taski_id).copy()
                # cal utility in taski coalition
                ui = cal_uav_utility_in_colition(
                    self,
                    taski,
                    self.uav_manager,
                    self.task_manager,
                    taski_coalition_copy,
                    self.hyper_params,
                )

            if taskj_id is None:
                taskj = None
                uj = 0
            else:
                taskj = self.task_manager.get(taskj_id)
                taskj_coalition_copy = self.coalition_manager.get_coalition(taskj_id).copy()
                # cal utility in taskj coalition
                # print(f"taskj_coalition_copy: {taskj_coalition_copy}")
                uj = cal_uav_utility_in_colition(
                    self,
                    taskj,
                    self.uav_manager,
                    self.task_manager,
                    taskj_coalition_copy,
                    self.hyper_params,
                )

            if ui < uj:
                # uav leave taski, join taskj
                self.coalition_manager.unassign(self.id)
                self.coalition_manager.assign(self.id, taskj.id if taskj is not None else None)
                divert_changed = True
                self.uav_update_step_dict[self.id] += 1
                print(f"uav: {self.id}, divert_changed: {divert_changed}")
        self.changed = self.changed or divert_changed
        print(f"[divert] uav: {self.id}, changed: {self.changed}")
        return divert_changed


from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def get_connected_components(uav_list: List[UAV], comm_distance: float) -> List[List[int]]:
    uav_nums = len(uav_list)
    distance_matrix = np.zeros((uav_nums, uav_nums))
    for ridx in range(uav_nums):
        for cidx in range(uav_nums):
            distance_matrix[ridx, cidx] = uav_list[ridx].position.distance_to(
                uav_list[cidx].position
            )

    neighbor_mask = distance_matrix < comm_distance
    neighbor_distance_matrix = distance_matrix * neighbor_mask

    graph = csr_matrix(neighbor_distance_matrix)
    n_components, labels = connected_components(graph, directed=False)
    components = [[] for _ in range(n_components)]
    for uav_idx, label in enumerate(labels):
        components[label].append(uav_list[uav_idx].id)
    # return components, by uav.id, not uav in list index
    return components


class ICRA2024_CoalitionFormationGame(MRTASolver):
    """
    1. based on cur coalition, try to divert to another task coaliton.
        1.1. if changed, update_step += 1, changed = True.
        1.2. if not changed, update_step, changed = False.
        self.uav_satisfied_dict[self.id] = changed
    2. self.send_msg() -> Message:
    3. self.receive(msgs):
        ```
        for msg in msgs:
            for uav_id in self.uav_ids:
                if self.uav_update_step_dict[uav_id] < msg.update_step:
                    # update
                    self.coaliton_manager.unassign(uav_id)
                    self.coalition_manager.assign(msg.uav2task[uav_id])
                    self.changed = True
        ```
    4. back to 1

    if all self.uav_satisfied_dict[self.id] == True:
        break the simulation.

    ```
    // Alg1: Decision-Making Alg for each ri

    Partition Alg1(R /*a set of I robots*/, T /*a set of J tasks*/, D /*a set of neighbors of the robots */) {
        // Initialize:
        // r_satisfied = 0;
        // ξi = 0;
        // Πi = {S0 = R, Sj = [] ∀tj ∈ T};
        while(r_satisfied == true) {
            if(r_satisfied_i == 0) {
            // Based on Equation (2), calculate the utility to
            // determine the Sj∗ and tj∗ that maximize its
            // utility;
            if ui(tj*, Sj*) > ui(txx, Sxx) {
                // joinSj*, update Πi
                ξi = ξi + 1;
            }
            r_satisfied_i = 1;
            }
            // Broadcast Mi = {Πi, r_satisfied, ξi} and
            // receive Mk from its neighbors Di
            // Collect all the messages and construct
            // Mi_all = {Mi, Mk}
            for(each message Mk in Mi_all) {
                if (ξk >= ξi) {
                    Mi = Mk;
                    if (Πi != Πk) r_satisfied_i = 0;
                }
            }
        }
        if (r_satisfied == 1) return Π0; //?
    }
    ```
    """

    def init_allocate(self, components: List[List[int]], debug=False):
        """
        ```
        for component in connected_components:
            for uav_id in component:
                uav: AutoUAV = self.uav_manager.get(uav_id)
                uav.init(component, self.task_manager)
        ```
        """

        # init auto uav
        for component in components:
            # component_uav_list = [self.uav_manager.get(uav_id) for uav_id in component]
            component_uav_manager = UAVManager(
                [self.uav_manager.get(uav_id) for uav_id in component]
            )
            for uav_id in component:
                uav: AutoUAV = self.uav_manager.get(uav_id)
                uav.init(component_uav_manager, self.task_manager, self.hyper_params)

    def run_allocate(self, debug=False):
        """
        每个uav是独立的个体，自己运行一套算法。
        中控模拟程序只需要触发每个uav的动作，模拟uav之间的通信等。
        """
        communication_distance = 15

        uav_list = self.uav_manager.get_all()
        components = get_connected_components(uav_list, communication_distance)
        print(f"components: {components}")

        self.init_allocate(components)
        iter_cnt = 0

        while True:  # sim step
            print(f"iter {iter_cnt}")
            if iter_cnt > self.hyper_params.max_iter:
                print(f"reach max iter {self.hyper_params.max_iter}")
                break

            total_changed = False

            for component in components:
                messages: List[Message] = []
                component_changed = False
                for uav_id in component:
                    uav: AutoUAV = self.uav_manager.get(uav_id)
                    uav.changed = False
                    uav.try_divert()
                    msg = uav.send_msg()
                    messages.append(msg)

                for uav_id in component:
                    uav: AutoUAV = self.uav_manager.get(uav_id)
                    uav.receive_and_update(messages)
                component_changed = any(msg.changed for msg in messages)
                total_changed = total_changed or component_changed
            
            print(f"total_changed: {total_changed}")
            if not total_changed:
                print("all uav not changed, break!")
                break

            iter_cnt += 1
        # remember to update the self.coalition_manager!
        for component in components:
            if len(component) == 0:
                continue
            leader_uav_id = component[0]
            leader_uav: AutoUAV = self.uav_manager.get(leader_uav_id)
            self.coalition_manager.merge_coalition_manager(leader_uav.coalition_manager)


if __name__ == "__main__":
    pass
