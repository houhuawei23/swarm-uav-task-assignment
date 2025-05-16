from typing import List, Tuple, Dict
from dataclasses import dataclass, field

import random
import numpy as np
from scipy.optimize import linear_sum_assignment

from framework.base import HyperParams, LogLevel
from framework.uav import UAV, UAVManager
from framework.task import Task, TaskManager
from framework.coalition_manager import CoalitionManager
from framework.mrta_solver import MRTASolver

# from framework.utils import calculate_obtained_resources, get_resources_weights
import framework.utils as utils

np.set_printoptions(precision=2)

log_level: LogLevel = LogLevel.SILENCE


def calculate_path_cost(
    uav: UAV, task: Task, resource_contribution: float, map_shape: Tuple, zero_resource_contribution_penalty: float = -1.0
):
    """Calculates the path cost for a UAV to reach a task.

    The path cost is computed based on the Euclidean distance between the UAV's current position and
    the task's position. The cost is normalized by the maximum possible distance in the environment.
    If the distance exceeds the maximum distance, the UAV is considered unable to reach the task.

    range: [0, 1], 
        0 means uav_to_task_distance is max,
        1 means uav_to_task_distance is zero.
    the larger the cost, the nearer the UAV is to the task, the better.

    当 val(ui, tj) <= 0 时, 设计 rui (tj) 小于 0.
    含义是当无人机 ui 加入任务 tj 联盟无法 贡献资源时, 加入该联盟的收益小于在 ct0 中的收益

    Args:
        uav (UAV): The UAV object representing the unmanned aerial vehicle.
        task (Task): The task object representing the task to be completed.

    Returns:
        float: The normalized path cost (between 0 and 1) if the task is reachable, otherwise -1.
    """
    # 路径成本计算
    if resource_contribution <= 0:
        return zero_resource_contribution_penalty

    # distance = np.linalg.norm(uav.position - task.position)
    distance = uav.position.distance_to(task.position)
    # max_distance = np.sqrt(100**2 + 100**2)  # 假设最大距离
    max_distance = np.linalg.norm(map_shape)  # 任务环境区域大小

    return 1 - distance / max_distance


def calculate_threat_cost(uav: UAV, task: Task):
    """Calculates the threat cost for a UAV assigned to a task.

    The threat cost is computed as the product of the UAV's intrinsic value and the task's threat index.
    This represents the risk or potential loss associated with assigning the UAV to the task.

    Args:
        uav (UAV): The UAV object representing the unmanned aerial vehicle.
        task (Task): The task object representing the task to be completed.

    Returns:
        float: The threat cost value.
    """
    # 威胁代价计算
    return uav.value * task.threat


def calculate_resource_contribution(
    uav: UAV,
    task: Task,
    task_collation: List[int],
    task_obtained_resources: List[float],
) -> float:
    """Calculates the resource contribution of a UAV to a task.

    task_obtained_resources = [...]
    now_required_resources = [1, 2, -3]
    np.maximum(now_required_resources, 0) = [1, 2, 0]
    uav.resources = [2, 3, 4]
    uav.resources - np.maximum(now_required_resources, 0) = [0, 1, 4]
    surplus_resources_sum_of_uav = 0 + 1 + 4 = 5

    uav 加入 task 的 task_coalition 的资源贡献，是其边际贡献。
    即，如果当前 task_coaliton 已经完全满足 task 的资源需求，那么 uav 加入该 coalition 的贡献为 0。
    """
    # 资源贡献计算
    # I = uav.resources + satisfied, paper meaning this???

    # +is required, -is surplus
    task_obtained_resources_copy = task_obtained_resources.copy()

    # 如果 uav 在 task 的联盟中，则减去 uav 的贡献
    if uav.id in task_collation:
        task_obtained_resources_copy -= uav.resources

    now_required_resources = task.required_resources - task_obtained_resources_copy
    surplus_resources_sum_of_uav = sum(
        np.maximum(uav.resources - np.maximum(now_required_resources, 0), 0)
    )

    # 表征 task 对资源的偏好
    Kj = utils.get_resources_weights(task.required_resources, task_obtained_resources_copy)
    I = uav.resources  # only consider uav's own resources
    O = surplus_resources_sum_of_uav
    P = 0.5

    if log_level >= LogLevel.DEBUG:
        print(f"Kj: {Kj}, I: {I}, O: {O: .2f}, P: {P: .2f}")
    # val = calculate_resource_contribution(Kj, I, O, P)
    """
    Returns:
        float: The resource contribution value.

    val(ui, tj) = K[j, :] · I - P · O
        K: (m, l) 权重矩阵, P: 常数权值, 对未利用资源的惩罚系数
        K[j, :] = [k1, k2, ..., kl]: 第 j 个任务的资源权重向量, K[j, k] 表示第 k 个资源的权重
        I = [i1, i2, ..., il]: 无人机加入任务对应的联盟集合中可利用的每类资源的数目, 取决于任务本身与联盟内其他成员?
        O: 该无人机未利用的总资源数目, 即加入该联盟后冗余资源总数目
    """
    contribution = np.dot(Kj, I) - P * O
    return max(0, contribution)


def calculate_uav_benefit_4_join_task_coalition(
    uav: UAV,
    task: Task,
    task_collation: List[int],  # may have or not have uav
    task_obtained_resources: List[float],
    hyper_params: HyperParams,
) -> float:
    """Calculates the total benefit of assigning a UAV to a task.

    O(1)

    r(ui, tj) = alpha * val(ui, tj) + path_cost_weight * cost(ui, tj) - threat_loss_weight * risk(ui, tj)

    The total benefit is a weighted sum of resource contribution, path cost, and threat cost.
    The weights (alpha, path_cost_weight, threat_loss_weight) are used to balance the importance of each factor.

    # epsilon(uav, task) =
    """
    if log_level >= LogLevel.DEBUG:
        print(f"cal_uav_task_benefit(uav=u{uav.id}, task=t{task.id})")
    # 计算资源贡献
    resource_contribution = calculate_resource_contribution(
        uav, task, task_collation, task_obtained_resources
    )
    # 计算路径成本
    path_cost = calculate_path_cost(
        uav, task, resource_contribution, hyper_params.map_shape, hyper_params.zero_resource_contribution_penalty
    )
    # 计算威胁代价
    threat_cost = calculate_threat_cost(uav, task)
    # 总收益
    total_benefit = (
        hyper_params.resource_contribution_weight * resource_contribution
        + hyper_params.path_cost_weight * path_cost
        - hyper_params.threat_loss_weight * threat_cost
    )
    # check time window
    min_uav_fly_time = uav.position.distance_to(task.position) / uav.max_speed
    if min_uav_fly_time > task.time_window[1]:
        total_benefit = -100
    # debug = True
    if log_level >= LogLevel.DEBUG:
        print(
            f"  val={resource_contribution:.2f}, path_cost={path_cost:.2f}, threat_cost={threat_cost:.2f}, benefit={total_benefit:.2f}"
        )
        print("cal_uav_task_benefit finished")
    return total_benefit


def calculate_task_benefit_4_given_coalition(
    task: Task,
    given_coalition: List[int],
    uav_manager: UAVManager,
    hyper_params: HyperParams,
    # debug=False,
) -> float:
    """
    Complexity: O(coalition.size()) -> max O(n)

    the benefit of a coalition for task = sum of the benefit of each UAV in the coalition.
    R(ctj) = sum(cal_benefit(ui, tj)) for ui in ctj
    """
    # if debug:
    if log_level >= LogLevel.DEBUG:
        print(f"cal_task_coalition_utility(task=t{task.id}, coalition={given_coalition})")
    utility = 0.0
    obtained_resources = utils.calculate_obtained_resources(
        given_coalition, uav_manager, hyper_params.resources_num
    )
    for uav_id in given_coalition:  # ??? coalition=[u1, u2] on t2, 效用不应该简单叠加吧！！
        if uav_id not in uav_manager.get_ids():
            # jump the uav not in uav_manager
            # print(f"uav_id={uav_id} not in uav_manager")
            continue
        uav = uav_manager.get(uav_id)

        benefit = calculate_uav_benefit_4_join_task_coalition(
            uav, task, given_coalition, obtained_resources, hyper_params
        )  # O(1)
        utility += benefit
    # if debug:
    if log_level >= LogLevel.DEBUG:
        print(f"  utility={utility:.2f}")
        print("cal_task_coalition_utility finished")
    return utility


def cal_uav_utility_in_colition(
    uav: UAV, task: Task, coalition: List[int], uav_manager: UAVManager, hyper_params: HyperParams
) -> float:
    """
    Complexity: O(n)
    calcualte utility of uav in coalition of task, dont change anything.
    if uav in coalition of task, utility = u_have - u_not_have
    if uav not in coalition of task, utility = u_have - u_not_have (will add uav to coalition to calculate)
    """

    coalition_copy = coalition.copy()
    if uav.id in coalition:  # uav already in coalition
        coalition_copy.remove(uav.id)

    u_not_have = calculate_task_benefit_4_given_coalition(
        task, coalition_copy, uav_manager, hyper_params
    )  # max: O(n)
    coalition_copy.append(uav.id)
    u_have = calculate_task_benefit_4_given_coalition(
        task, coalition_copy, uav_manager, hyper_params
    )
    utility = u_have - u_not_have
    return utility


@dataclass
class ChinaScience2024_CoalitionFormationGame(MRTASolver):
    """Represents a coalition formation game where UAVs are assigned to tasks based on benefits.

    The game involves forming coalitions of UAVs for each task, considering factors such as resource contribution,
    path cost, and threat. The goal is to maximize the overall benefit of the assignments.

    G = (U, E, epsilon, R)

    where:
        U: is a set of UAVs.
        E: is the task set selected by the UAV, equivalent to the task set T.
        epsiolon: is a utility function that evaluates the benefits of a drone.
        R: is a function that evaluates the utility of individual coalitions.

    R(ctj) 任务 tj 的联盟效用: R(ctj) = sum_{ui in ctj} [r(ui, tj)], 即执行该任务的所有无人机的效用之和。
    SR 任务分配问题的总收益: SR = sum_{ctj in CS} R(ctj)
    ui 执行 tj 的收益: r(ui, tj) = alpha * val(ui, tj) + path_cost_weight * dist(ui, tj) - threat_loss_weight * risk(ui, tj)
    epsilon(eu, E_{-ui}) 无人机效用函数: epsilon(eu, E_{-ui}) = R(ctj) - R(ctj - {ui})

    Game Goal: 最大化总收益 SR
    约束条件:
        (1) 速度约束;
        (2) 任务时效性约束;
        (3) 无人机执行任务模式约束: 同一时间只允许一个无人机执行一个任务。
    """

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
        return "CSCI2024_Xue"

    def cal_benefit_matrix(self, unassigned_uav_ids: List[int], task_ids: List[int]) -> np.ndarray:
        """Calculates the benefit matrix for UAVs and tasks. 计算每个无人机对每个任务的收益矩阵

        O(n*m)

        The benefit matrix is a 2D array where each element represents the benefit of assigning a specific UAV to a specific task.
        The benefit is calculated based on resource contribution, path cost, and threat cost.

        Returns:
            np.ndarray: A 2D numpy array of shape (n, m), where n is the number of UAVs and m is the number of tasks.
                        Each element benefit_matrix[i, j] represents the benefit of assigning UAV i to task j.
        """
        n = len(unassigned_uav_ids)
        m = len(task_ids)
        benefit_matrix = np.zeros((n, m))

        for i, uav_id in enumerate(unassigned_uav_ids):  # n
            for j, task_id in enumerate(task_ids):  # m
                uav = self.uav_manager.get(uav_id)
                task = self.task_manager.get(task_id)
                coalition = self.coalition_manager.get_coalition(task_id)
                obtained_resources = utils.calculate_obtained_resources(
                    coalition, self.uav_manager, self.hyper_params.resources_num
                )  # O(n)

                benefit_matrix[i, j] = calculate_uav_benefit_4_join_task_coalition(
                    uav, task, coalition, obtained_resources, self.hyper_params
                )  # O(1)
        if log_level >= LogLevel.DEBUG:
            print("Benefit Matrix:")
            print(benefit_matrix)
        return benefit_matrix

    def cal_sum_utility(self) -> float:
        """Calculates the total utility of the current coalition assignments.
        SR = sum_{ctj in CS} R(ctj)
        """
        total_utility = 0.0
        for task in self.task_manager:
            total_utility += calculate_task_benefit_4_given_coalition(
                task,
                self.coalition_manager.get_coalition(task.id),
                self.uav_manager,
                self.hyper_params,
            )
        if log_level >= LogLevel.DEBUG:
            print(f"Total Utility: {total_utility:.2f}")
        return total_utility

    def match_tasks(self, benefit_matrix, unassigned_uav_ids, task_ids):
        """Matches UAVs to tasks based on the benefit matrix.

        O(n^3) + O(m)

        This function uses the Hungarian algorithm (linear sum assignment) to find the optimal assignment
        of UAVs to tasks that maximizes the total benefit. UAVs are assigned to tasks only if the benefit
        is positive. The coalitions dictionary is updated to reflect the assignments.

        Algorithm Steps:
        1. Compute the benefit matrix using `calculate_benefit_matrix`.
        2. Use the Hungarian algorithm to find the optimal assignment.
        3. Assign UAVs to tasks if the benefit is positive.
        4. Update the coalitions dictionary to reflect the assignments.

        每个任务匹配到一个无人机
        """
        # 最大加权匹配, Complexity: worst case O(n^3), average case near O(n^2)
        row_ind, col_ind = linear_sum_assignment(benefit_matrix, maximize=True)  # O(n^3) ?
        # print(f"row_ind: {row_ind}, col_ind: {col_ind}")
        have_assigned = False
        for uav_idx, task_idx in zip(row_ind, col_ind):  # O(m)
            # print(unassigned_uav_ids, task_ids)
            # print(f"uav_idx: {uav_idx}, task_idx: {task_idx}")
            # print("here", unassigned_uav_ids[uav_idx])
            uav: UAV = self.uav_manager.get(unassigned_uav_ids[uav_idx])
            task: Task = self.task_manager.get(task_ids[task_idx])
            if log_level >= LogLevel.DEBUG:
                print(f"uav_idx: {uav_idx}, task_idx: {task_idx}")

            if benefit_matrix[uav_idx, task_idx] > 0:
                if log_level >= LogLevel.INFO:
                    print(f"Assigning u{uav.id} to t{task.id}")
                # 6. update the required resources of task
                # task_obtained_resources += uav.resources
                self.coalition_manager.assign(uav.id, task.id)
                have_assigned = True
        return have_assigned

    def check_stability(self):
        """Checks the stability of the current coalition assignments.

        Complexity: O(n^2)

        A coalition is considered stable if no UAV can improve its benefit by switching to another task.
        This function iterates through all UAVs and tasks to check if any UAV can achieve a higher benefit
        by being reassigned to a different task. If such a case is found, the UAV is reassigned, and the
        coalition is marked as unstable.

        Returns:
            bool: True if the coalition is stable, False otherwise.
        """
        stable = True
        # 遍历所有任务的所有已分配的无人机
        for taski in self.task_manager:
            taski_coalition = self.coalition_manager.get_coalition(taski.id)
            for uav_id in taski_coalition:  # max O(n)
                uav = self.uav_manager.get(uav_id)
                cur_utility = cal_uav_utility_in_colition(
                    uav, taski, taski_coalition.copy(), self.uav_manager, self.hyper_params
                )

                if log_level >= LogLevel.DEBUG:
                    print(f"Cur utility: u{uav.id}-t{taski.id}={cur_utility}")

                for taskj in self.task_manager:  # O(n): 至多遍历所有的无人机
                    if taskj.id != taski.id:
                        taskj_coalition = self.coalition_manager.get_coalition(taskj.id)

                        move_to_taskj_utility = cal_uav_utility_in_colition(
                            uav, taskj, taskj_coalition.copy(), self.uav_manager, self.hyper_params
                        )

                        if log_level >= LogLevel.DEBUG:
                            print(f"utility: u{uav.id}-t{taskj.id}={move_to_taskj_utility}")
                        if move_to_taskj_utility > cur_utility:
                            self.coalition_manager.unassign(uav.id)  # 无人机退出原联盟
                            stable = False
                            break
        if log_level >= LogLevel.INFO:
            print(f"check_stability {stable}")

        return stable

    def run_allocate(self):
        """Runs the coalition formation game until a stable assignment is achieved.

        This function repeatedly matches UAVs to tasks and checks for stability until no further
        improvements can be made. The process terminates when the coalition assignments are stable.

        Returns:
            dict: The final coalitions dictionary mapping task IDs to lists of assigned UAVs.

        1. Initial task allocation result.
        2. Calculate or update the benefit matric.
        3. Matching based on maximum weighed principle.
        4. Update the required resources of task, design rules for profit checking.
        5. Obtain the final stable result of this layer.
        6. Update the carried resources and requirements of tasks.
        7. Requirements of a certain task are an empty set.
            1. if true, jump to 2.
            2. else go to 8.
        8. Exit.When all tasks exit, obtain the final coalition structure.

        max_iter x [O(n m) + O(n^2 + m) + O(n^2)]
        max_iter x [O(n m) + O(n^2)]
        max_iter x [O(n m) + O(n^2)]
        """
        iter_cnt = 0
        while True:  # max_iter cnt
            # in once iter, try to assign one uav to each task
            if log_level >= LogLevel.INFO:
                print(f"Iteration {iter_cnt} begin.")
                print(f"Cur coalition set: {self.coalition_manager}")
            if iter_cnt >= self.hyper_params.max_iter:
                if log_level >= LogLevel.DEBUG:
                    print(f"Max iterations {iter_cnt} reached, may have dead loop")
                break  # break while

            iter_cnt += 1
            # 2. calculate the benefit matrix
            unassigned_uav_ids = self.coalition_manager.get_unassigned_uav_ids().copy()
            task_ids = self.task_manager.get_ids()

            benefit_matrix = self.cal_benefit_matrix(unassigned_uav_ids, task_ids)  # O(n*m)

            # 3. matching based on maximum weighed principle (based on benefit matrix)
            if not self.match_tasks(benefit_matrix, unassigned_uav_ids, task_ids):  # O(n^2 + m)
                # TODO: 问题，根据收益矩阵为每个任务选择使得总最大的无人机；
                # 问题是如果所有无人机对某任务的收益都小于 0, 按理来说不应该给该任务分配任何的无人机
                # 此处算法实现是否会导致给该任务分配一个收益 < 0 的无人机？
                # print("No more UAVs can be assigned to tasks. Over, break.")
                break

            self.check_stability()  # O(n^2)

        return self.coalition_manager
