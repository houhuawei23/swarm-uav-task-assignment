from typing import List, Tuple, Dict, Type
import random
import numpy as np
from itertools import combinations
from math import factorial
from dataclasses import dataclass, field
from copy import deepcopy
from scipy.optimize import linear_sum_assignment


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

from .utils import MRTA_CFG_Model, get_connected_components_uavid, MRTA_CFG_Model_HyperParams

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
    # 知道自己的临近无人机的id uav_ids => 知道临近uav的所有信息
    # 知道全局的任务(包括任务信息)
    # uav_manager: UAVManager = field(default=None, init=False)
    # task_manager: TaskManager = field(default=None, init=False)
    # coalition_manager: CoalitionManager = field(default=None, init=False)
    # hyper_params: HyperParams = field(default=None, init=False)
    """

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
        map_shape: List[float] = utils.calculate_map_shape_on_mana(uav_manager, task_manager)
        max_distance = max(map_shape)
        max_uav_value = max(uav.value for uav in uav_manager.get_all())
        self.model_hparams = MRTA_CFG_Model_HyperParams(
            max_distance=max_distance,
            max_uav_value=max_uav_value,
            w_sat=50.0,
            w_waste=1,
            w_dist=2,
            w_threat=1,
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
        for msg in msgs:  # O(component.size())
            # print(f"msg: {msg}")
            for msg_uav_id, msg_uav_update_step in msg.uav_update_step_dict.items():  # O(n)
                if msg_uav_id not in self.uav_update_step_dict:
                    self.uav_update_step_dict[msg_uav_id] = 0
                if self.uav_update_step_dict[msg_uav_id] < msg_uav_update_step:
                    self.coalition_manager.unassign(msg_uav_id)
                    self.coalition_manager.assign(msg_uav_id, msg.uav2task[msg_uav_id])
                    self.uav_update_step_dict[msg_uav_id] = msg_uav_update_step
                    receive_changed = True

        self.changed = self.changed or receive_changed
        return receive_changed

    def try_divert(self, prefer: str = "cooperative") -> bool:
        """
        Complexity: O(n m)
        """
        # pass
        task_ids = self.task_manager.get_ids().copy()
        # task_ids.append(TaskManager.free_uav_task_id)  # special, 0 means no task
        # print(f"u{self.id}.changed: {self.changed}")
        divert_changed = False
        for taskj_id in task_ids:  # m
            taski_id = self.coalition_manager.get_taskid(self.id)
            # print(f"u{self.id}.try_divert: taski_id: {taski_id}, taskj_id: {taskj_id}")
            # self.coalition_manager.format_print()  # check
            if taski_id == taskj_id:
                continue

            taski = self.task_manager.get(taski_id)
            taskj = self.task_manager.get(taskj_id)

            prefer_func = MRTA_CFG_Model.get_prefer_func(prefer)
            if prefer_func(
                uav=self,
                task_p=taski,
                task_q=taskj,
                uav_manager=self.uav_manager,
                task_manager=self.task_manager,
                coalition_manager=self.coalition_manager,
                resources_num=self.hyper_params.resources_num,
                model_hparams=self.model_hparams,
            ):
                # print(f"divert changed: {self.id} leave {taski_id}, join {taskj_id}")
                # if true, uav leave taski, join taskj

                self.coalition_manager.unassign(self.id)
                self.coalition_manager.assign(self.id, taskj_id)
                divert_changed = True
                self.uav_update_step_dict[self.id] += 1  # !!!!
                # self.coalition_manager.format_print()  # check
                # break  # if uav changed task, break, next uav
        # print(f"u{self.id}.changed: {self.changed}, divert_changed: {divert_changed}")
        self.changed = self.changed or divert_changed
        # print(f"u{self.id}.changed: {self.changed}")
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


from termcolor import colored, cprint


class DistributedSolver(MRTASolver):
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
        map_shape: List[float] = utils.calculate_map_shape_on_mana(uav_manager, task_manager)
        max_distance = max(map_shape)
        max_uav_value = max(uav.value for uav in uav_manager.get_all())
        self.model_hparams = MRTA_CFG_Model_HyperParams(
            max_distance=max_distance,
            max_uav_value=max_uav_value,
            w_sat=95.0,
            w_waste=5,
            w_dist=35,
            w_threat=10,
        )

    @staticmethod
    def uav_type() -> Type:
        return AutoUAV

    @classmethod
    def type_name(cls):
        return "Distributed"

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
        task_ids = self.task_manager.get_ids()
        init_assignment: Dict[int, List[int]] = {
            task_id: [] for task_id in task_ids
        }  # taskid -> uavids
        # init_assignment = {0: [3], 1: [2], 2: [1]}
        for component in components:
            for uav_id in component:
                # print(f"uav_id: {uav_id}")
                task_id = random.choice(task_ids)
                # if task_id not in init_assignment.keys():
                #     init_assignment[task_id] = []
                init_assignment[task_id].append(uav_id)
        # init_assignment = {1: [], 2: [1, 2, 3], 0: []}  #!!!
        # init auto uav
        for component in components:
            # component_uav_list = [self.uav_manager.get(uav_id) for uav_id in component]
            component_uav_manager = UAVManager(
                [self.uav_manager.get(uav_id) for uav_id in component]
            )
            for uav_id in component:
                uav: AutoUAV = self.uav_manager.get(uav_id)
                uav_coalition_manager = CoalitionManager(self.uav_manager.get_ids(), task_ids)
                uav.init(
                    component_uav_manager,
                    self.task_manager,
                    uav_coalition_manager,
                    self.hyper_params,
                )
                uav_coalition_manager.update_from_assignment(init_assignment, uav.uav_manager)
                # task_id = random.choice(task_ids)
                # uav.coalition_manager.assign(uav_id, task_id)

    def init_allocate_beta(self, components: List[List[int]], debug=False):
        """
        分布式初始化分配方法。
        对每个连通分量（component）独立执行类似于集中式的最大权匹配分配策略。

        每个component中：
        1. 每一轮迭代中，给每个任务分配一个未经初始化分配的无人机
        2. 计算所有未分配UAV加入各任务联盟的边际收益，形成收益矩阵
        3. 使用最大权匹配算法优化当前轮次的分配
        4. 迭代直至该component中所有UAV都被分配

        Args:
            components: List[List[int]] - UAV id的连通分量列表
            debug: bool - 是否打印调试信息

        Returns:
            bool: 分配是否成功
        """
        task_ids = [
            tid for tid in self.task_manager.get_ids() if tid != TaskManager.free_uav_task_id
        ]

        # 初始化每个component的分配结果
        init_assignment = {task_id: [] for task_id in self.task_manager.get_ids()}

        # 对每个component单独执行分配
        for component in components:
            # 创建该component的UAV管理器
            component_uav_manager = UAVManager(
                [self.uav_manager.get(uav_id) for uav_id in component]
            )

            # 初始化该component的coalition manager
            component_coalition_manager = CoalitionManager(component, self.task_manager.get_ids())

            # 获取当前component中未分配的UAV
            unassigned_uavs = component_uav_manager.get_all()

            # 当还有未分配的UAV时，继续迭代
            while unassigned_uavs:
                # 构建收益矩阵
                benefit_matrix = np.zeros((len(unassigned_uavs), len(task_ids)))

                # 计算每个未分配UAV对每个任务的边际收益
                for i, uav in enumerate(unassigned_uavs):
                    for j, task_id in enumerate(task_ids):
                        task = self.task_manager.get(task_id)
                        # 获取当前任务的联盟
                        current_coalition = component_coalition_manager.get_coalition(task_id)
                        current_coalition_uavs = [
                            component_uav_manager.get(uid) for uid in current_coalition
                        ]

                        # 计算当前联盟的效用
                        before = MRTA_CFG_Model.cal_coalition_eval(
                            task,
                            current_coalition_uavs,
                            self.hyper_params.resources_num,
                            model_hparams=self.model_hparams,
                        )

                        # 计算加入新UAV后的效用
                        new_coalition_uavs = current_coalition_uavs + [uav]
                        after = MRTA_CFG_Model.cal_coalition_eval(
                            task,
                            new_coalition_uavs,
                            self.hyper_params.resources_num,
                            model_hparams=self.model_hparams,
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
                        component_coalition_manager.assign(uav.id, task_id)
                        # 更新全局分配结果
                        init_assignment[task_id].append(uav.id)
                    else:
                        # 如果收益为负，分配到free_uav_task
                        component_coalition_manager.assign(uav.id, TaskManager.free_uav_task_id)
                        init_assignment[TaskManager.free_uav_task_id].append(uav.id)

                    removed_uavs.append(uav)

                # 更新未分配的UAV列表
                unassigned_uavs = [uav for uav in unassigned_uavs if uav not in removed_uavs]

            # 为该component中的每个UAV初始化其局部视图
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
                # 更新UAV的局部coalition manager
                uav_coalition_manager.update_from_assignment(init_assignment, uav.uav_manager)

        return True

    def run_allocate(self):
        self._run_allocate(init_method="beta", prefer="cooperative")

    def _run_allocate(self, init_method: str = "beta", prefer: str = "cooperative"):
        """
        每个uav是独立的个体，自己运行一套算法。
        中控模拟程序只需要触发每个uav的动作，模拟uav之间的通信等。
        """
        # comm_distance = self.hyper_params.comm_distance
        comm_distance = self.hyper_params.map_shape[0]
        uav_list = self.uav_manager.get_all()
        components = get_connected_components_uavid(uav_list, comm_distance)  # max: O(n + n^2)
        # cprint(f"components: {components}", "green")

        if init_method == "beta":
            self.init_allocate_beta(components)  # O(n)
        elif init_method == "random":
            self.init_allocate(components)  # O(n)

        if log_level >= LogLevel.DEBUG:
            for uav in uav_list:
                print(uav.debug_info())
        max_not_cahnged_iter = 5
        not_changed_iter_cnt = 0
        iter = 0

        sample_rate = 1 / 3
        rec_max_not_changed_iter = int(1 / sample_rate) + 10  # 期望来看，每个uav都会被抽样到

        # Comlexity: max_iter x O(n^2 m)
        while True:  # sim step
            iter += 1
            # print(f"iter {iter}")
            if not_changed_iter_cnt > max_not_cahnged_iter:
                if log_level >= LogLevel.INFO:
                    print(f"reach max_not_cahnged_iter {max_not_cahnged_iter}")
                break
            if not_changed_iter_cnt > rec_max_not_changed_iter:
                if log_level >= LogLevel.INFO:
                    print(f"reach rec_max_not_changed_iter {rec_max_not_changed_iter}")
                break
            if iter > self.hyper_params.max_iter:
                if log_level >= LogLevel.INFO:
                    print(f"reach max iter {self.hyper_params.max_iter}")
                break
            total_changed = False
            # Complexity:
            # n x sample_rate x O(n m) + O(n x component.size() x n)
            # max: O(n^2 m) + O(n^3)
            for component in components:
                # Warning: if not random sample, may be deadlock!!! vibrate!!!
                rec_sample_size = max(1, int(len(component) * sample_rate))
                sampled_uavids = random.sample(component, rec_sample_size)
                # sampled_uavids = component
                messages: List[Message] = []
                component_changed = False
                # for uav_id in component:
                # print(f"component: {component}, sampled_uavids: {sampled_uavids}")
                for uav_id in sampled_uavids:
                    # print(f"uav_id: {uav_id}")
                    uav: AutoUAV = self.uav_manager.get(uav_id)
                    # print(uav.debug_info())
                    uav.changed = False
                    uav.try_divert(prefer=prefer)  # random, O(n m)
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
                leader_uav_id = component[0]
                leader_uav: AutoUAV = self.uav_manager.get(leader_uav_id)
                msg = leader_uav.send_msg()
                leader_messages.append(msg)

            # Complexity: components.size() x components.size() x n
            for component in components:  # components.size()
                if len(component) == 0:
                    continue
                leader_uav_id = component[0]
                leader_uav: AutoUAV = self.uav_manager.get(leader_uav_id)
                leader_uav.receive_and_update(leader_messages)

            leaders_changed = any(msg.changed for msg in leader_messages)
            total_changed = total_changed or leaders_changed

            if log_level >= LogLevel.INFO:
                print(f"iter {not_changed_iter_cnt} end")
                # self.coalition_manager.format_print()  # check
                for component in components:
                    for uav_id in component:
                        uav: AutoUAV = self.uav_manager.get(uav_id)
                        print(uav.debug_info())
                print("--------------------------------")

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
            leader_uav_id = component[0]
            leader_uav: AutoUAV = self.uav_manager.get(leader_uav_id)
            self.coalition_manager.merge_coalition_manager(leader_uav.coalition_manager)

        if log_level >= LogLevel.INFO:
            print("final coalition:")
            self.coalition_manager.format_print()  # check


class DistributedSolver_Selfish(DistributedSolver):
    @classmethod
    def type_name(cls):
        return "Distributed_Selfish"

    def run_allocate(self):
        self._run_allocate(init_method="beta", prefer="selfish")


class DistributedSolver_Pareto(DistributedSolver):
    @classmethod
    def type_name(cls):
        return "Distributed_Pareto"

    def run_allocate(self):
        self._run_allocate(init_method="beta", prefer="pareto")


class DistributedSolver_RandomInit(DistributedSolver):
    @classmethod
    def type_name(cls):
        return "Distributed_RandomInit"

    def run_allocate(self):
        self._run_allocate(init_method="random")
