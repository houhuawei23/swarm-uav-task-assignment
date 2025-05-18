from typing import List, Tuple, Dict, Type
import random
import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy

from framework.base import HyperParams, LogLevel
from framework.uav import UAV, UAVManager
from framework.task import Task, TaskManager
from framework.coalition_manager import CoalitionManager
from framework.mrta_solver import MRTASolver

# from .iros2024 import cal_uav_utility_in_colition
from . import iros2024
from . import csci2024

from .utils import get_connected_components_uavid


log_level: LogLevel = LogLevel.SILENCE


@dataclass
class Message:
    uav_id: int
    changed: bool
    uav_update_step_dict: Dict[int, int]
    task2coalition: Dict[int, List[int]]
    uav2task: Dict[int, int]


@dataclass
class AutoUAV(UAV):
    """
    r(tj, Sj^ck) = vj * log(min(wj^ck, |Sj^ck|) + e, wj^ck)

    ui(tj, Sj) = (Sum[ck in C]{ r(tj, Sj^ck) }) / |Sj^ck| - cost(i, j)

    cost(i, j) = distance(i, j)
    """

    # 知道自己的临近无人机的id uav_ids => 知道临近uav的所有信息
    # 知道全局的任务(包括任务信息)
    # uav_manager: UAVManager = field(default=None, init=False)
    # task_manager: TaskManager = field(default=None, init=False)
    # coalition_manager: CoalitionManager = field(default=None, init=False)
    # hyper_params: HyperParams = field(default=None, init=False)

    changed: bool = field(default=False, init=False)
    uav_update_step_dict: Dict[int, int] = field(default=None, init=False)

    def __init__(
        self,
        id: int,
        position: List[float] | np.ndarray,
        resources: List[float] | np.ndarray,
        value: float,
        max_speed: float,
        mass: float | None = 1.0,
        fly_energy_per_time: float = random.uniform(1, 3),
        hover_energy_per_time: float = random.uniform(1, 3),
    ):
        super().__init__(
            id,
            position,
            resources,
            value,
            max_speed,
            mass,
            fly_energy_per_time,
            hover_energy_per_time,
        )

    def __post_init__(self):
        super().__post_init__()

    def init(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
    ):
        self.uav_manager = uav_manager
        self.task_manager = task_manager
        # self.coalition_manager = CoalitionManager(uav_manager.get_ids(), task_manager.get_ids())
        self.coalition_manager = coalition_manager
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
            uav_update_step_dict=deepcopy(self.uav_update_step_dict),
            task2coalition=deepcopy(self.coalition_manager.get_task2coalition()),
            uav2task=deepcopy(self.coalition_manager.get_uav2task()),
        )
        return msg

    def receive_and_update(self, msgs: List[Message]) -> bool:
        """
        Complexity: O(component.size() x n), max: O(n^2)

        receive msgs from its neighbors, update local coalition.
        """
        receive_changed = False
        
        # Track all changes to apply them atomically
        changes_to_apply = []
        
        for msg in msgs:
            for msg_uav_id, msg_uav_update_step in msg.uav_update_step_dict.items():
                if msg_uav_id not in self.uav_update_step_dict:
                    self.uav_update_step_dict[msg_uav_id] = 0
                if self.uav_update_step_dict[msg_uav_id] < msg_uav_update_step:
                    changes_to_apply.append((msg_uav_id, msg.uav2task[msg_uav_id], msg_uav_update_step))
        
        # Apply changes atomically to maintain consistency
        for uav_id, task_id, update_step in changes_to_apply:
            self.coalition_manager.unassign(uav_id)
            self.coalition_manager.assign(uav_id, task_id)
            self.uav_update_step_dict[uav_id] = update_step
            receive_changed = True
        
        self.changed = self.changed or receive_changed
        return receive_changed

    def try_divert(self) -> bool:
        """
        Complexity: O(n m)
        """
        # pass
        task_ids = self.task_manager.get_ids().copy()
        task_ids.append(TaskManager.free_uav_task_id)  # special, 0 means no task
        divert_changed = False
        for taskj_id in task_ids:  # m
            taski_id = self.coalition_manager.get_taskid(self.id)
            # print(f"taski_id: {taski_id}, taskj_id: {taskj_id}")
            if taski_id == taskj_id:
                continue

            if taski_id == TaskManager.free_uav_task_id:
                ui = 0
            else:
                taski = self.task_manager.get(taski_id)
                taski_coalition_copy = self.coalition_manager.get_coalition(taski_id).copy()
                # cal utility in taski coalition
                # ui = iros2024.cal_uav_utility_in_colition(
                #     self,
                #     taski,
                #     taski_coalition_copy,
                #     self.uav_manager,
                #     self.task_manager,
                #     self.hyper_params,
                # )
                ui = csci2024.cal_uav_utility_in_colition(
                    self,
                    taski,
                    taski_coalition_copy,
                    self.uav_manager,
                    self.hyper_params,
                )  # O(n)
                # print(f"taski_coalition_copy: {taski_coalition_copy}")
            if taskj_id == TaskManager.free_uav_task_id:
                uj = 0
            else:
                taskj = self.task_manager.get(taskj_id)
                taskj_coalition_copy = self.coalition_manager.get_coalition(taskj_id).copy()
                # cal utility in taskj coalition
                # print(f"taskj_coalition_copy: {taskj_coalition_copy}")
                # uj = iros2024.cal_uav_utility_in_colition(
                #     self,
                #     taskj,
                #     taskj_coalition_copy,
                #     self.uav_manager,
                #     self.task_manager,
                #     self.hyper_params,
                # )
                uj = csci2024.cal_uav_utility_in_colition(
                    self,
                    taskj,
                    taskj_coalition_copy,
                    self.uav_manager,
                    self.hyper_params,
                )
                # print(f"taskj_coalition_copy: {taskj_coalition_copy}")

            if ui < uj:
                # uav leave taski, join taskj
                self.coalition_manager.unassign(self.id)
                self.coalition_manager.assign(self.id, taskj_id)
                divert_changed = True
                self.uav_update_step_dict[self.id] += 1
                # print(f"uav: {self.id}, divert_changed: {divert_changed}")
        self.changed = self.changed or divert_changed
        # print(f"[divert] uav: {self.id}, changed: {self.changed}")
        return divert_changed

    def brief_info(self):
        info = f"AU_{self.id}(re={self.resources}, val={self.value}, spd={self.max_speed})"
        return info

    def debug_info(self):
        info = f"AU_{self.id}(re={self.resources}, pos={self.position.tolist()}, val={self.value}, spd={self.max_speed})\n"
        info += f"  uavs: {self.uav_manager.brief_info()}\n"
        info += f"  tsks: {self.task_manager.brief_info()}\n"
        info += f"  coalitoin: {self.coalition_manager.brief_info()}\n"
        info += f"  update_dict: {self.uav_update_step_dict}\n"
        return info


class ICRA2024_CoalitionFormationGame(MRTASolver):
    """
    TODO: 该让不同component的无人机组之间能相互通信，通过leader uav...
    否则每一组无人机都是孤立进行任务分配，整体效能太差

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

    def __init__(
        self,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
    ):
        super().__init__(uav_manager, task_manager, coalition_manager, hyper_params)

    @staticmethod
    def uav_type() -> Type:
        return AutoUAV

    @classmethod
    def type_name(cls):
        return "ICRA2024_LiwangZhang"

    def init_allocate(self, components: List[List[int]], debug=False):
        """
        Complexity: O(n)
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
                uav_coalition_manager = CoalitionManager(
                    self.uav_manager.get_ids(), self.task_manager.get_ids()
                )
                uav.init(
                    component_uav_manager,
                    self.task_manager,
                    uav_coalition_manager,
                    self.hyper_params,
                )

    def run_allocate(self):
        """
        每个uav是独立的个体，自己运行一套算法。
        中控模拟程序只需要触发每个uav的动作，模拟uav之间的通信等。
        """
        # comm_distance = self.hyper_params.comm_distance
        comm_distance = self.hyper_params.map_shape[0] / 3
        uav_list = self.uav_manager.get_all()
        components = get_connected_components_uavid(uav_list, comm_distance)  # max: O(n + n^2)
        # print(f"components: {components}")

        self.init_allocate(components)  # O(n)
        if log_level >= LogLevel.DEBUG:
            for uav in uav_list:
                print(uav.debug_info())

        not_changed_iter_cnt = 0
        sample_rate = 1 / 3
        rec_max_iter = int(1 / sample_rate) + 1  # 期望来看，每个uav都会被抽样到

        # Comlexity: max_iter x O(n^2 m)
        while True:  # sim step
            # print(f"iter {iter_cnt}")
            if (
                not_changed_iter_cnt > self.hyper_params.max_iter
                or not_changed_iter_cnt > rec_max_iter
            ):
                # print(f"reach max iter {self.hyper_params.max_iter}")
                break

            total_changed = False
            # Complexity:
            # n x sample_rate x O(n m) + O(n x component.size() x n)
            # max: O(n^2 m) + O(n^3)
            for component in components:
                # Warning: if not random sample, may be deadlock!!! vibrate!!!
                rec_sample_size = max(1, int(len(component) * sample_rate))
                sampled_uavids = random.sample(component, rec_sample_size)
                messages: List[Message] = []
                component_changed = False
                # for uav_id in component:
                for uav_id in sampled_uavids:
                    uav: AutoUAV = self.uav_manager.get(uav_id)
                    uav.changed = False
                    uav.try_divert()  # random, O(n m)
                    msg = uav.send_msg()
                    messages.append(msg)
                    if log_level >= LogLevel.DEBUG:
                        print(uav.debug_info())

                for uav_id in component:  # O(component.size()^2 x n)
                    uav: AutoUAV = self.uav_manager.get(uav_id)
                    uav.receive_and_update(messages)  # max: O(n^2)
                    if log_level >= LogLevel.DEBUG:
                        print(uav.debug_info())

                component_changed = any(msg.changed for msg in messages)
                total_changed = total_changed or component_changed
            # TODO: leader follower, leader communication
            leader_messages: List[Message] = []  # component.size()
            for component in components:
                if len(component) == 0:
                    continue
                leader_uav_id = self.select_leader(component, self.uav_manager)
                leader_uav: AutoUAV = self.uav_manager.get(leader_uav_id)
                msg = leader_uav.send_msg()
                leader_messages.append(msg)

            # Complexity: components.size() x components.size() x n
            for component in components:  # components.size()
                if len(component) == 0:
                    continue
                leader_uav_id = self.select_leader(component, self.uav_manager)
                leader_uav: AutoUAV = self.uav_manager.get(leader_uav_id)
                leader_uav.receive_and_update(leader_messages)

            leaders_changed = any(msg.changed for msg in leader_messages)
            total_changed = total_changed or leaders_changed

            # print(f"total_changed: {total_changed}")
            if not total_changed:
                if log_level >= LogLevel.INFO:
                    print("all uav not changed, break!")
                break

            not_changed_iter_cnt += 1
        # remember to update the self.coalition_manager!
        for component in components:
            if len(component) == 0:
                continue
            # leader_uav_id = component[0]
            leader_uav_id = self.select_leader(component, self.uav_manager)
            leader_uav: AutoUAV = self.uav_manager.get(leader_uav_id)
            self.coalition_manager.merge_coalition_manager(leader_uav.coalition_manager)

    def select_leader(self, component, uav_manager):
        if not component:
            return None
        
        # Simple approach: select UAV closest to the center of the component
        positions = [uav_manager.get(uav_id).position.xyz for uav_id in component]
        center = np.mean(positions, axis=0)
        
        min_dist = float('inf')
        leader_id = None
        
        for uav_id in component:
            uav = uav_manager.get(uav_id)
            dist = np.linalg.norm(uav.position.xyz - center)
            if dist < min_dist:
                min_dist = dist
                leader_id = uav_id
            
        return leader_id


class ICRA2024_CoalitionFormationGame_2(ICRA2024_CoalitionFormationGame):
    @classmethod
    def type_name(cls):
        return "ICRA2024_LiwangZhang2"

    def run_allocate(self):
        """
        每个uav是独立的个体，自己运行一套算法。
        中控模拟程序只需要触发每个uav的动作，模拟uav之间的通信等。
        """
        # comm_distance = self.hyper_params.comm_distance
        comm_distance = self.hyper_params.map_shape[0] / 3
        uav_list = self.uav_manager.get_all()
        components = get_connected_components_uavid(uav_list, comm_distance)  # max: O(n + n^2)
        # print(f"components: {components}")

        self.init_allocate(components)  # O(n)
        if log_level >= LogLevel.DEBUG:
            for uav in uav_list:
                print(uav.debug_info())

        iter_cnt = 0
        sample_rate = 1 / 3
        rec_max_iter = int(1 / sample_rate) + 1  # 期望来看，每个uav都会被抽样到

        # Comlexity: max_iter x O(n^2 m)
        while True:  # sim step
            # print(f"iter {iter_cnt}")
            if iter_cnt > self.hyper_params.max_iter or iter_cnt > rec_max_iter:
                # print(f"reach max iter {self.hyper_params.max_iter}")
                break

            total_changed = False
            # Complexity:
            # n x sample_rate x O(n m) + O(n x component.size() x n)
            # max: O(n^2 m) + O(n^3)
            for component in components:
                # Warning: if not random sample, may be deadlock!!! vibrate!!!
                rec_sample_size = max(1, int(len(component) * sample_rate))
                sampled_uavids = random.sample(component, rec_sample_size)
                messages: List[Message] = []
                component_changed = False
                # for uav_id in component:
                for uav_id in sampled_uavids:
                    uav: AutoUAV = self.uav_manager.get(uav_id)
                    uav.changed = False
                    uav.try_divert()  # random, O(n m)
                    msg = uav.send_msg()
                    messages.append(msg)
                    if log_level >= LogLevel.DEBUG:
                        print(uav.debug_info())

                for uav_id in component:  # O(component.size()^2 x n)
                    uav: AutoUAV = self.uav_manager.get(uav_id)
                    uav.receive_and_update(messages)  # max: O(n^2)
                    if log_level >= LogLevel.DEBUG:
                        print(uav.debug_info())

                component_changed = any(msg.changed for msg in messages)
                total_changed = total_changed or component_changed
            # TODO: leader follower, leader communication
            leader_messages: List[Message] = []  # component.size()
            for component in components:
                if len(component) == 0:
                    continue
                leader_uav_id = self.select_leader(component, self.uav_manager)
                leader_uav: AutoUAV = self.uav_manager.get(leader_uav_id)
                msg = leader_uav.send_msg()
                leader_messages.append(msg)

            # Complexity: components.size() x components.size() x n
            for component in components:  # components.size()
                if len(component) == 0:
                    continue
                leader_uav_id = self.select_leader(component, self.uav_manager)
                leader_uav: AutoUAV = self.uav_manager.get(leader_uav_id)
                leader_uav.receive_and_update(leader_messages)

            leaders_changed = any(msg.changed for msg in leader_messages)
            total_changed = total_changed or leaders_changed

            # print(f"total_changed: {total_changed}")
            if not total_changed:
                if log_level >= LogLevel.INFO:
                    print("all uav not changed, break!")
                break

            iter_cnt += 1
        # remember to update the self.coalition_manager!
        for component in components:
            if len(component) == 0:
                continue
            leader_uav_id = self.select_leader(component, self.uav_manager)
            leader_uav: AutoUAV = self.uav_manager.get(leader_uav_id)
            self.coalition_manager.merge_coalition_manager(leader_uav.coalition_manager)


if __name__ == "__main__":
    pass
