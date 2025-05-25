from typing import List, Tuple, Dict, Type
from dataclasses import dataclass
import random
import numpy as np

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

from itertools import combinations
from math import factorial

log_level = LogLevel.SILENCE


# Problem Model
## Coalition Structure Evaluation
### Task Resources Satisfaction
class Activations:
    @staticmethod
    def zero_one(req: float, obt: float) -> float:
        if req == 0:
            return 1
        return 0 if obt <= req else 1

    @staticmethod
    def linear(req: float, obt: float) -> float:
        if req == 0:
            return 1
        return (obt / req) if obt <= req else 1

    @staticmethod
    def quadratic(req: float, obt: float) -> float:
        if req == 0:
            return 1
        return (obt / req) ** 2 if obt <= req else 1

    @staticmethod
    def poly(req: float, obt: float, n: int = 5) -> float:
        if req == 0:
            return 1
        return (obt / req) ** n if obt <= req else 1

    @staticmethod
    def sigmoid(req: float, obt: float) -> float:
        if req == 0:
            return 1
        return 1 / (1 + np.exp(-obt / req))

    @staticmethod
    def tanh(req: float, obt: float) -> float:
        if req == 0:
            return 1
        return np.tanh(obt / req)

    @staticmethod
    def relu(req: float, obt: float) -> float:
        if req == 0:
            return 1
        return max(0, obt / req)

    @staticmethod
    def leaky_relu(req: float, obt: float, alpha: float = 0.01) -> float:
        if req == 0:
            return 1
        return max(alpha * obt / req, obt / req)

    @staticmethod
    def elu(req: float, obt: float, alpha: float = 1.0) -> float:
        if req == 0:
            return 1
        return alpha * (np.exp(obt / req) - 1) if obt / req < 0 else obt / req

    @staticmethod
    def get_all() -> List[callable]:
        return [
            Activations.zero_one,
            Activations.linear,
            Activations.quadratic,
            Activations.poly,
            # Activations.sigmoid,
            # Activations.tanh,
            # Activations.relu,
            # Activations.leaky_relu,
            # Activations.elu,
        ]

    @staticmethod
    def plot_all():
        import matplotlib.pyplot as plt

        o_v = np.linspace(0, 1.4, 100)
        plt.figure(figsize=(10, 6))
        for act in Activations.get_all():
            y = [act(1, o) for o in o_v]
            plt.plot(o_v, y, label=act.__name__)

        plt.legend()
        plt.grid()
        plt.show()


def eval_res_sat(req: float, obt: float, type: str = "poly") -> float:
    res = 0.0
    activation_types = ["zero_one", "linear", "quadratic", "poly"]

    if type not in activation_types:
        raise ValueError(f"activation type {type} not in {activation_types}")
    if type == "zero_one":
        res = Activations.zero_one(req, obt)
    elif type == "linear":
        res = Activations.linear(req, obt)
    elif type == "quadratic":
        res = Activations.quadratic(req, obt)
    elif type == "poly":
        res = Activations.poly(req, obt)

    # check
    if res < 0 or res > 1:
        raise ValueError(f"res_sat {res} not in [0, 1]")

    return res


### UAV Resources Waste
def eval_res_waste(req: float, obt: float) -> float:
    if obt <= req:
        return 0
    else:
        return (obt - req) / (req + 1)


@dataclass
class MRTA_CFG_Model_HyperParams:
    max_distance: float = 100.0
    max_uav_value: float = 20.0
    w_sat: float = 1.0
    w_waste: float = 1.0
    w_dist: float = 1.0
    w_threat: float = 1.0


class MRTA_CFG_Model:
    @staticmethod
    def eval_res_sat(t: Task, c: List[UAV], resources_num: int) -> float:
        # if t.id == TaskManager.free_uav_task_id:
        #     return 0.0
        if c is None or len(c) == 0:
            return 0.0
        # sat = sum(eval_res_sat(t.required_resources[i], uav.resources[i], type="poly") for i in range(resources_num) for uav in c)
        req_v = t.required_resources
        obt_v = sum(uav.resources for uav in c)
        # print(req_v, obt_v)
        # sat = sum(eval_res_sat(req_v[i], obt_v[i], type="poly") for i in range(resources_num)) / resources_num
        sat = 0.0
        for i in range(resources_num):
            # print(i, eval_res_sat(req_v[i], obt_v[i], type="poly"))
            sat += eval_res_sat(req_v[i], obt_v[i], type="poly")
        sat /= resources_num
        return sat

    @staticmethod
    def eval_is_complete(t: Task, c: List[UAV], resources_num: int) -> bool:
        if c is None or len(c) == 0:
            return False
        req_v = t.required_resources
        obt_v = sum(uav.resources for uav in c)
        return all(req_v[i] <= obt_v[i] for i in range(resources_num))

    @staticmethod
    def eval_res_waste(t: Task, c: List[UAV], resources_num: int) -> float:
        if c is None or len(c) == 0:
            return 0.0
        req_v = t.required_resources
        obt_v = sum(uav.resources for uav in c)
        return sum(eval_res_waste(req_v[i], obt_v[i]) for i in range(resources_num)) / resources_num

    @staticmethod
    def eval_dist_cost(t: Task, c: List[UAV], max_distance: float = 100.0) -> float:
        if c is None or len(c) == 0:
            return 0.0
        return sum(u.position.distance_to(t.position) for u in c) / max_distance

    @staticmethod
    def eval_threat_cost(t: Task, c: List[UAV], max_uav_value: float = 20.0) -> float:
        if c is None or len(c) == 0:
            return 0.0
        return sum(u.value * t.threat for u in c) / max_uav_value

    @staticmethod
    def cal_task_comp_index(
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
        weight_v: np.array = None,
    ) -> float:
        if weight_v is None:
            weight_v: np.ndarray = np.ones(task_manager.size())
            weight_v /= weight_v.sum()
        index = 0.0
        # for t in task_manager.get_all():
        task_list = task_manager.get_all()
        for t_idx, t in enumerate(task_list):
            w = weight_v[t_idx]
            coalition_uav_ids = coalition_manager.get_coalition(t.id)
            coalition_uavs = [uav_manager.get(uav_id) for uav_id in coalition_uav_ids]
            # print("ee", coalition_uav_ids, coalition_uavs)
            index += w * MRTA_CFG_Model.eval_res_sat(t, coalition_uavs, hyper_params.resources_num)
        return index

    @staticmethod
    def cal_res_sat_index(
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
        weight_v: np.array = None,
    ) -> float:
        if weight_v is None:
            weight_v: np.ndarray = np.ones(task_manager.size())
            weight_v /= weight_v.sum()
        index = 0.0
        task_list = task_manager.get_all()
        # for t in task_manager.get_all():
        for t_idx, t in enumerate(task_list):
            w = weight_v[t_idx]
            coalition_uav_ids = coalition_manager.get_coalition(t.id)
            coalition_uavs = [uav_manager.get(uav_id) for uav_id in coalition_uav_ids]
            index += w * MRTA_CFG_Model.eval_res_sat(t, coalition_uavs, hyper_params.resources_num)
        return index

    @staticmethod
    def cal_res_waste_index(
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
        weight_v: np.array = None,
    ) -> float:
        if weight_v is None:
            weight_v: np.ndarray = np.ones(task_manager.size())
            weight_v /= weight_v.sum()
        index = 0.0
        task_list = task_manager.get_all()
        # for t in task_manager.get_all():
        for t_idx, t in enumerate(task_list):
            w = weight_v[t_idx]
            coalition_uav_ids = coalition_manager.get_coalition(t.id)
            coalition_uavs = [uav_manager.get(uav_id) for uav_id in coalition_uav_ids]
            index += w * MRTA_CFG_Model.eval_res_waste(
                t, coalition_uavs, hyper_params.resources_num
            )
        return index

    @staticmethod
    def cal_dist_cost_index(
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
        weight_v: np.array = None,
        model_hyper_params: MRTA_CFG_Model_HyperParams = None,
    ) -> float:
        if weight_v is None:
            weight_v: np.ndarray = np.ones(task_manager.size())
            weight_v /= weight_v.sum()
        index = 0.0
        task_list = task_manager.get_all()
        # for t in task_manager.get_all():
        for t_idx, t in enumerate(task_list):
            w = weight_v[t_idx]
            coalition_uav_ids = coalition_manager.get_coalition(t.id)
            coalition_uavs = [uav_manager.get(uav_id) for uav_id in coalition_uav_ids]
            index += w * MRTA_CFG_Model.eval_dist_cost(
                t, coalition_uavs, model_hyper_params.max_distance
            )
        return index

    @staticmethod
    def cal_threat_cost_index(
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
        weight_v: np.array = None,
        model_hyper_params: MRTA_CFG_Model_HyperParams = None,
    ) -> float:
        if weight_v is None:
            weight_v: np.ndarray = np.ones(task_manager.size())
            weight_v /= weight_v.sum()
        index = 0.0
        task_list = task_manager.get_all()
        # for t in task_manager.get_all():
        for t_idx, t in enumerate(task_list):
            w = weight_v[t_idx]
            coalition_uav_ids = coalition_manager.get_coalition(t.id)
            coalition_uavs = [uav_manager.get(uav_id) for uav_id in coalition_uav_ids]
            index += w * MRTA_CFG_Model.eval_threat_cost(
                t, coalition_uavs, model_hyper_params.max_uav_value
            )
        return index

    @staticmethod
    def cal_coalition_eval(
        t: Task,
        c: List[UAV],
        resources_num: int,
        model_hparams: MRTA_CFG_Model_HyperParams,
    ) -> float:
        if t.id == TaskManager.free_uav_task_id:
            return 0.0
        # 如何确定权重？？？？
        # 单位不同！！需要归一化，如何归一化？？？
        w_sat = model_hparams.w_sat
        w_waste = model_hparams.w_waste
        w_dist = model_hparams.w_dist
        w_threat = model_hparams.w_threat
        res_sat = w_sat * MRTA_CFG_Model.eval_res_sat(t, c, resources_num)
        res_waste = w_waste * MRTA_CFG_Model.eval_res_waste(t, c, resources_num)
        dist_cost = w_dist * MRTA_CFG_Model.eval_dist_cost(t, c, model_hparams.max_distance)
        threat_cost = w_threat * MRTA_CFG_Model.eval_threat_cost(t, c, model_hparams.max_uav_value)
        # print(
        #     f"res_sat: {res_sat}, res_waste: {res_waste}, dist_cost: {dist_cost}, threat_cost: {threat_cost}"
        # )
        result = res_sat - res_waste - dist_cost - threat_cost
        return result

    @staticmethod
    def cal_cs_eval_beta(
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
        weight_v: np.ndarray = None,
        # weight_v,
    ) -> float:
        # print("\ncal_cs_eval_beta:")
        if weight_v is None:
            weight_v = np.ones(task_manager.size())
            weight_v /= weight_v.sum()
        eval_result = 0.0
        task_list = task_manager.get_all()
        # for t in task_manager.get_all():
        for t_idx, t in enumerate(task_list):
            w = weight_v[t_idx]
            coalition_uav_ids = coalition_manager.get_coalition(t.id)
            coalition_uavs = [uav_manager.get(uav_id) for uav_id in coalition_uav_ids]
            cur = w * MRTA_CFG_Model.cal_coalition_eval(
                t, coalition_uavs, hyper_params.resources_num
            )
            # print(f"cur task {t.id}", cur)
            eval_result += cur
        return eval_result

    @staticmethod
    def cal_cs_eval(
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        hyper_params: HyperParams,
    ) -> float:
        # print("\ncal_cs_eval:")
        w_sat = 1.0
        w_waste = 0.1
        w_dist = 0.1
        w_threat = 0.1
        res_sat_index_w = w_sat * MRTA_CFG_Model.cal_res_sat_index(
            uav_manager, task_manager, coalition_manager, hyper_params
        )
        res_waste_index_w = w_waste * MRTA_CFG_Model.cal_res_waste_index(
            uav_manager, task_manager, coalition_manager, hyper_params
        )
        dist_cost_index_w = w_dist * MRTA_CFG_Model.cal_dist_cost_index(
            uav_manager, task_manager, coalition_manager, hyper_params
        )
        threat_cost_index_w = w_threat * MRTA_CFG_Model.cal_threat_cost_index(
            uav_manager, task_manager, coalition_manager, hyper_params
        )
        print(f"res_sat_index_w: {res_sat_index_w}")
        print(f"res_waste_index_w: {res_waste_index_w}")
        print(f"dist_cost_index_w: {dist_cost_index_w}")
        print(f"threat_cost_index_w: {threat_cost_index_w}")
        eval_result = res_sat_index_w - res_waste_index_w - dist_cost_index_w - threat_cost_index_w
        return eval_result

    @staticmethod
    def cal_coalition_payoff_shapley_value(
        task: Task,
        coalition: List[UAV],
        resources_num: int,
        model_hparams: MRTA_CFG_Model_HyperParams = MRTA_CFG_Model_HyperParams(),
    ) -> Dict:
        """
        cal coalition payoff dict, using Shapley Value.
        Problem: Computational Complexity is too high.
            O(2^n) Exponential
        """
        coalition_size = len(coalition)
        phi = np.zeros(coalition_size)

        def v(S: List[UAV]) -> float:
            return MRTA_CFG_Model.cal_coalition_eval(
                task, S, resources_num, model_hparams=model_hparams
            )

        # 计算每个玩家的 Shapley 值
        for i in range(coalition_size):
            for k in range(coalition_size):
                for S in combinations(range(coalition_size), k):
                    if i not in S:
                        S = set(S)
                        marginal_contribution = v([coalition[j] for j in (S | {i})]) - v(
                            [coalition[j] for j in S]
                        )
                        phi[i] += (
                            factorial(len(S))
                            * factorial(coalition_size - len(S) - 1)
                            / factorial(coalition_size)
                        ) * marginal_contribution
        uavid_to_phi = {uav.id: phi[i] for i, uav in enumerate(coalition)}
        return uavid_to_phi

    @staticmethod
    def cal_uav_benefit(
        uav: UAV,
        task: Task,
        coalition: List[UAV],
        resources_num: int,
        model_hparams: MRTA_CFG_Model_HyperParams,
        method: str = "monte_carlo",
    ) -> float:
        return MRTA_CFG_Model.cal_coalition_payoff(
            task, coalition, resources_num, method=method, model_hparams=model_hparams
        )[uav.id]

    # Preference
    @staticmethod
    def selfish_prefer(
        uav: UAV,
        task_p: Task,
        task_q: Task,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        resources_num: int,
        model_hparams: MRTA_CFG_Model_HyperParams,
    ) -> bool:
        """
        uav prefer task_q than task_p
        """
        # print(f"selfish_prefer: uav {uav.id}: from u{task_p.id} to u{task_q.id}")
        coalition_tp_uavids = coalition_manager.get_coalition(task_p.id)
        if uav.id not in coalition_tp_uavids:
            raise Exception("uav not in coalition")
        coalition_tp_uavs = [uav_manager.get(uav_id) for uav_id in coalition_tp_uavids]
        benefit_tp = MRTA_CFG_Model.cal_uav_benefit(
            uav,
            task_p,
            coalition_tp_uavs,
            resources_num,
            model_hparams=model_hparams,
            method="monte_carlo",
        )

        coalition_tq_uavids = coalition_manager.get_coalition(task_q.id)
        coalition_tq_uavs = [uav_manager.get(uav_id) for uav_id in coalition_tq_uavids]
        coalition_tq_uavs.append(uav)
        benefit_tq = MRTA_CFG_Model.cal_uav_benefit(
            uav,
            task_q,
            coalition_tq_uavs,
            resources_num,
            model_hparams=model_hparams,
            method="monte_carlo",
        )
        # print(f"benefit_tp: {benefit_tp}")
        # print(f"benefit_tq: {benefit_tq}")
        if benefit_tq > benefit_tp:
            return True
        else:
            return False

    @staticmethod
    def pareto_prefer(
        uav: UAV,
        task_p: Task,
        task_q: Task,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        resources_num: int,
        model_hparams: MRTA_CFG_Model_HyperParams,
    ) -> bool:
        """
        uav prefer task_q than task_p
        """
        # print(f"pareto_prefer: uav {uav.id}: from t{task_p.id} to t{task_q.id}")
        csa_payoff_dict = dict()
        csb_payoff_dict = dict()
        coalition_tp_uavids = coalition_manager.get_coalition(task_p.id)
        if uav.id not in coalition_tp_uavids:
            raise Exception("uav not in coalition")
        coalition_tp_uavs = [uav_manager.get(uav_id) for uav_id in coalition_tp_uavids]
        csa_coalition_tp_payoff = MRTA_CFG_Model.cal_coalition_payoff(
            task_p,
            coalition_tp_uavs,
            resources_num,
            method="monte_carlo",
            model_hparams=model_hparams,
        )
        csa_payoff_dict.update(csa_coalition_tp_payoff)

        coalition_tq_uavids = coalition_manager.get_coalition(task_q.id)
        coalition_tq_uavs = [uav_manager.get(uav_id) for uav_id in coalition_tq_uavids]
        csa_coalition_tq_payoff = MRTA_CFG_Model.cal_coalition_payoff(
            task_q,
            coalition_tq_uavs,
            resources_num,
            method="monte_carlo",
            model_hparams=model_hparams,
        )
        csa_payoff_dict.update(csa_coalition_tq_payoff)

        influnenced_uavs = coalition_tp_uavs + coalition_tq_uavs

        # divert
        coalition_tp_uavs.remove(uav)
        coalition_tq_uavs.append(uav)
        csb_coalition_tp_payoff = MRTA_CFG_Model.cal_coalition_payoff(
            task_p,
            coalition_tp_uavs,
            resources_num,
            method="monte_carlo",
            model_hparams=model_hparams,
        )
        csb_payoff_dict.update(csb_coalition_tp_payoff)
        csb_coalition_tq_payoff = MRTA_CFG_Model.cal_coalition_payoff(
            task_q,
            coalition_tq_uavs,
            resources_num,
            method="monte_carlo",
            model_hparams=model_hparams,
        )
        csb_payoff_dict.update(csb_coalition_tq_payoff)
        # print(f"csa_payoff_dict: {csa_payoff_dict}")
        # print(f"csb_payoff_dict: {csb_payoff_dict}")
        # compare
        if all(csb_payoff_dict[u.id] >= csa_payoff_dict[u.id] for u in influnenced_uavs) and (
            csb_payoff_dict[uav.id] > csa_payoff_dict[uav.id]
        ):
            return True
        else:
            return False

    @staticmethod
    def cooperative_prefer(
        uav: UAV,
        task_p: Task,
        task_q: Task,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        resources_num: int,
        model_hparams: MRTA_CFG_Model_HyperParams,
    ) -> bool:
        """
        uav prefer task_q than task_p
        """
        csa_payoff_dict = dict()
        csb_payoff_dict = dict()
        coalition_tp_uavids = coalition_manager.get_coalition(task_p.id)
        if uav.id not in coalition_tp_uavids:
            raise Exception("uav not in coalition")
        coalition_tp_uavs = [uav_manager.get(uav_id) for uav_id in coalition_tp_uavids]
        csa_coalition_tp_payoff = MRTA_CFG_Model.cal_coalition_payoff(
            task_p, coalition_tp_uavs, resources_num
        )
        csa_payoff_dict.update(csa_coalition_tp_payoff)

        coalition_tq_uavids = coalition_manager.get_coalition(task_q.id)
        coalition_tq_uavs = [uav_manager.get(uav_id) for uav_id in coalition_tq_uavids]
        csa_coalition_tq_payoff = MRTA_CFG_Model.cal_coalition_payoff(
            task_q, coalition_tq_uavs, resources_num
        )
        csa_payoff_dict.update(csa_coalition_tq_payoff)

        influnenced_uavs = coalition_tp_uavs + coalition_tq_uavs
        # divert
        coalition_tp_uavs.remove(uav)
        coalition_tq_uavs.append(uav)

        csb_coalition_tp_payoff = MRTA_CFG_Model.cal_coalition_payoff(
            task_p, coalition_tp_uavs, resources_num
        )
        csb_payoff_dict.update(csb_coalition_tp_payoff)
        csb_coalition_tq_payoff = MRTA_CFG_Model.cal_coalition_payoff(
            task_q, coalition_tq_uavs, resources_num
        )
        csb_payoff_dict.update(csb_coalition_tq_payoff)
        # print(f"csa_payoff_dict: {csa_payoff_dict}")
        # print(f"csb_payoff_dict: {csb_payoff_dict}")
        # compare

        # according to shapley value,
        if sum(csb_payoff_dict.values()) >= sum(csa_payoff_dict.values()):
            return True
        else:
            return False

    @staticmethod
    def cooperative_prefer_beta(
        uav: UAV,
        task_p: Task,
        task_q: Task,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        resources_num: int,
        model_hparams: MRTA_CFG_Model_HyperParams,
    ) -> bool:
        """
        uav: task_p -> task_q
        uav prefer task_q than task_p

        because of the Efficiency property of Shapley Value, have:
            - sum(csb_payoff_dict.values()) = csb_coalition_tp_eval + csb_coalition_tq_eval
            - sum(csa_payoff_dict.values()) = csa_coalition_tp_eval + csa_coalition_tq_eval

        can directly compare the sum of payoff, to decide whether to divert.
        """
        if log_level >= LogLevel.INFO:
            print(f"cooperative_prefer_beta: uav {uav.id}: from t{task_p.id} to t{task_q.id}")
        coalition_tp_uavids = coalition_manager.get_coalition(task_p.id)
        if uav.id not in coalition_tp_uavids:
            raise Exception("uav not in coalition")
        # coalition_tp_uavs = [uav_manager.get(uav_id) for uav_id in coalition_tp_uavids]
        coalition_tp_uavs: List[UAV] = []
        for uav_id in coalition_tp_uavids:
            if uav_id in uav_manager.get_ids():
                coalition_tp_uavs.append(uav_manager.get(uav_id))

        coalition_tq_uavids = coalition_manager.get_coalition(task_q.id)
        # coalition_tq_uavs = [uav_manager.get(uav_id) for uav_id in coalition_tq_uavids]
        coalition_tq_uavs: List[UAV] = []
        for uav_id in coalition_tq_uavids:
            if uav_id in uav_manager.get_ids():
                coalition_tq_uavs.append(uav_manager.get(uav_id))
        # before divert
        csa_coalition_tp_eval = MRTA_CFG_Model.cal_coalition_eval(
            task_p,
            coalition_tp_uavs,
            resources_num,
            model_hparams=model_hparams,
        )
        csa_coalition_tq_eval = MRTA_CFG_Model.cal_coalition_eval(
            task_q,
            coalition_tq_uavs,
            resources_num,
            model_hparams=model_hparams,
        )

        if log_level >= LogLevel.INFO:
            print(
                f"csa_coalition_tp_eval: {csa_coalition_tp_eval}, tp=t{task_p.id}, uavs={coalition_tp_uavids}"
            )
            print(
                f"csa_coalition_tq_eval: {csa_coalition_tq_eval}, tq=t{task_q.id}, uavs={coalition_tq_uavids}"
            )

        # divert
        coalition_tp_uavs.remove(uav)
        coalition_tq_uavs.append(uav)
        coalition_tp_uavids = [u.id for u in coalition_tp_uavs]
        coalition_tq_uavids = [u.id for u in coalition_tq_uavs]
        csb_coalition_tp_eval = MRTA_CFG_Model.cal_coalition_eval(
            task_p,
            coalition_tp_uavs,
            resources_num,
            model_hparams=model_hparams,
        )
        csb_coalition_tq_eval = MRTA_CFG_Model.cal_coalition_eval(
            task_q,
            coalition_tq_uavs,
            resources_num,
            model_hparams=model_hparams,
        )

        if log_level >= LogLevel.INFO:
            print(
                f"csb_coalition_tp_eval: {csb_coalition_tp_eval}, tp=t{task_p.id}, uavs={coalition_tp_uavids}"
            )
            print(
                f"csb_coalition_tq_eval: {csb_coalition_tq_eval}, tq=t{task_q.id}, uavs={coalition_tq_uavids}"
            )

        # compare
        if (
            csb_coalition_tp_eval + csb_coalition_tq_eval
            > csa_coalition_tp_eval + csa_coalition_tq_eval
        ):
            return True
        else:
            return False

    @staticmethod
    def cooperative_exchange_prefer(
        uavi: UAV,
        uavj: UAV,
        uav_manager: UAVManager,
        task_manager: TaskManager,
        coalition_manager: CoalitionManager,
        resources_num: int,
        model_hparams: MRTA_CFG_Model_HyperParams,
    ) -> bool:
        """
        uavi and uavj exchange their task

        Args:
            uavi: First UAV to exchange
            uavj: Second UAV to exchange
            uav_manager: Manager containing all UAVs
            task_manager: Manager containing all tasks
            coalition_manager: Manager tracking task assignments
            resources_num: Number of resource types
            model_hparams: Hyperparameters for the model

        Returns:
            bool: True if the exchange would be beneficial for the overall system
        """
        if log_level >= LogLevel.INFO:
            print(f"cooperative_exchange_prefer: uav {uavi.id} and uav {uavj.id} exchange tasks")

        # Get tasks for both UAVs
        task_i_id = coalition_manager.get_taskid(uavi.id)
        task_j_id = coalition_manager.get_taskid(uavj.id)
        
        if task_i_id is None or task_j_id is None:
            raise Exception("One or both UAVs are not assigned to any task")
            
        task_i = task_manager.get(task_i_id)
        task_j = task_manager.get(task_j_id)

        # Get coalitions for both tasks
        coalition_i_uavids = coalition_manager.get_coalition_by_uav_id(uavi.id)
        coalition_j_uavids = coalition_manager.get_coalition_by_uav_id(uavj.id)

        # Get UAV objects for both coalitions
        coalition_i_uavs: List[UAV] = []
        for uav_id in coalition_i_uavids:
            if uav_id in uav_manager.get_ids():
                coalition_i_uavs.append(uav_manager.get(uav_id))

        coalition_j_uavs: List[UAV] = []
        for uav_id in coalition_j_uavids:
            if uav_id in uav_manager.get_ids():
                coalition_j_uavs.append(uav_manager.get(uav_id))

        # Calculate coalition evaluations before exchange
        csa_coalition_i_eval = MRTA_CFG_Model.cal_coalition_eval(
            task_i,
            coalition_i_uavs,
            resources_num,
            model_hparams=model_hparams,
        )
        csa_coalition_j_eval = MRTA_CFG_Model.cal_coalition_eval(
            task_j,
            coalition_j_uavs,
            resources_num,
            model_hparams=model_hparams,
        )

        if log_level >= LogLevel.INFO:
            print(
                f"csa_coalition_i_eval: {csa_coalition_i_eval}, ti=t{task_i.id}, uavs={coalition_i_uavids}"
            )
            print(
                f"csa_coalition_j_eval: {csa_coalition_j_eval}, tj=t{task_j.id}, uavs={coalition_j_uavids}"
            )

        # Perform exchange
        coalition_i_uavs.remove(uavi)
        coalition_j_uavs.remove(uavj)
        coalition_i_uavs.append(uavj)
        coalition_j_uavs.append(uavi)

        # Calculate coalition evaluations after exchange
        csb_coalition_i_eval = MRTA_CFG_Model.cal_coalition_eval(
            task_i,
            coalition_i_uavs,
            resources_num,
            model_hparams=model_hparams,
        )
        csb_coalition_j_eval = MRTA_CFG_Model.cal_coalition_eval(
            task_j,
            coalition_j_uavs,
            resources_num,
            model_hparams=model_hparams,
        )

        if log_level >= LogLevel.INFO:
            print(
                f"csb_coalition_i_eval: {csb_coalition_i_eval}, ti=t{task_i.id}, uavs={[u.id for u in coalition_i_uavs]}"
            )
            print(
                f"csb_coalition_j_eval: {csb_coalition_j_eval}, tj=t{task_j.id}, uavs={[u.id for u in coalition_j_uavs]}"
            )

        # Compare total coalition values
        if (
            csb_coalition_i_eval + csb_coalition_j_eval
            > csa_coalition_i_eval + csa_coalition_j_eval
        ):
            return True
        else:
            return False

    @staticmethod
    def get_prefer_func(prefer: str = "cooperative") -> callable:
        if prefer == "selfish":
            return MRTA_CFG_Model.selfish_prefer
        elif prefer == "pareto":
            return MRTA_CFG_Model.pareto_prefer
        elif prefer == "cooperative":
            # return Centralized_MRTA_Model.cooperative_prefer
            return MRTA_CFG_Model.cooperative_prefer_beta

    @staticmethod
    def cal_coalition_payoff_shapley_monte_carlo(
        task: Task,
        coalition: List[UAV],
        resources_num: int,
        num_samples: int = 100,
        model_hparams: MRTA_CFG_Model_HyperParams = MRTA_CFG_Model_HyperParams(),
    ) -> Dict:
        """
        Calculate coalition payoff using Monte Carlo sampling approximation of Shapley values.
        This is more efficient than exact calculation for large coalitions.

        Args:
            task: The task to evaluate
            coalition: List of UAVs in the coalition
            resources_num: Number of resource types
            num_samples: Number of random permutations to sample (default: 1000)

        Returns:
            Dict mapping UAV IDs to their approximate Shapley values
        """
        coalition_size = len(coalition)
        if coalition_size == 0:
            return {}

        # Initialize Shapley values for each player
        phi = np.zeros(coalition_size)

        def v(S: List[UAV]) -> float:
            return MRTA_CFG_Model.cal_coalition_eval(
                task, S, resources_num, model_hparams=model_hparams
            )

        # Monte Carlo sampling
        for _ in range(num_samples):
            # Generate random permutation of player indices
            perm = np.random.permutation(coalition_size)

            # For each player, calculate marginal contribution in this permutation
            for i in range(coalition_size):
                player_idx = perm[i]
                # Players before current player in permutation
                predecessors = [coalition[perm[j]] for j in range(i)]
                # Add current player
                with_player = predecessors + [coalition[player_idx]]
                # Calculate marginal contribution
                marginal = v(with_player) - v(predecessors)
                phi[player_idx] += marginal

        # Average the contributions
        phi /= num_samples

        # Convert to dictionary mapping UAV IDs to values
        uavid_to_phi = {uav.id: phi[i] for i, uav in enumerate(coalition)}
        return uavid_to_phi

    # def cal_coalition_payoff_mar

    @staticmethod
    def cal_coalition_payoff(
        task: Task,
        coalition: List[UAV],
        resources_num: int,
        method: str = "exact",
        model_hparams: MRTA_CFG_Model_HyperParams = MRTA_CFG_Model_HyperParams(),
    ) -> Dict:
        if method == "exact":
            return MRTA_CFG_Model.cal_coalition_payoff_shapley_value(
                task, coalition, resources_num, model_hparams=model_hparams
            )
        elif method == "monte_carlo":
            return MRTA_CFG_Model.cal_coalition_payoff_shapley_monte_carlo(
                task, coalition, resources_num, num_samples=50, model_hparams=model_hparams
            )
        else:
            raise ValueError(f"Invalid method: {method}")


from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def get_connected_components(distance_matrix: np.ndarray, comm_distance: float) -> List[List[int]]:
    """
    max: O(n + n^2)
    """
    neighbor_mask = distance_matrix < comm_distance
    neighbor_distance_matrix = distance_matrix * neighbor_mask

    graph = csr_matrix(neighbor_distance_matrix)
    n_components, labels = connected_components(
        graph, directed=False
    )  # O(n + e), n is uav num, e is edge num
    components = [[] for _ in range(n_components)]
    for idx, label in enumerate(labels):
        components[label].append(idx)
    return components


def get_connected_components_old(uav_list: List[UAV], comm_distance: float) -> List[List[int]]:
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


def get_connected_components_uavid(uav_list: List[UAV], comm_distance: float) -> List[List[int]]:
    uav_nums = len(uav_list)
    distance_matrix = np.zeros((uav_nums, uav_nums))
    for ridx in range(uav_nums):
        for cidx in range(uav_nums):
            distance_matrix[ridx, cidx] = uav_list[ridx].position.distance_to(
                uav_list[cidx].position
            )
    components_in_idx = get_connected_components(distance_matrix, comm_distance)
    components_in_uav_id = []
    for component_in_idx in components_in_idx:
        component_in_uav_id = [uav_list[idx].id for idx in component_in_idx]
        components_in_uav_id.append(component_in_uav_id)

    return components_in_uav_id


def test_model_eval():
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
    t = task_list[0]
    model_hparams = MRTA_CFG_Model_HyperParams()
    sat = MRTA_CFG_Model.eval_res_sat(t, uav_list, uav_gen.resources_num)
    is_complete = MRTA_CFG_Model.eval_is_complete(t, uav_list, uav_gen.resources_num)
    res_waste = MRTA_CFG_Model.eval_res_waste(t, uav_list, uav_gen.resources_num)
    dist_cost = MRTA_CFG_Model.eval_dist_cost(t, uav_list, model_hparams.max_distance)
    threat_cost = MRTA_CFG_Model.eval_threat_cost(t, uav_list, model_hparams.max_uav_value)
    print("sat", sat)
    print("is_complete", is_complete)
    print("res_waste", res_waste)
    print("dist_cost", dist_cost)
    print("threat_cost", threat_cost)


def test_model_index():
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

    t = task_list[0]
    model_hparams = MRTA_CFG_Model_HyperParams()
    coalition_manager = CoalitionManager(uav_manager.get_ids(), task_manager.get_ids())
    for uav in uav_list:
        coalition_manager.assign(uav.id, t.id)
    # coalition_manager.format_print()

    task_comp_index = MRTA_CFG_Model.cal_task_comp_index(
        uav_manager, task_manager, coalition_manager, hyper_params
    )
    res_sat_index = MRTA_CFG_Model.cal_res_sat_index(
        uav_manager, task_manager, coalition_manager, hyper_params
    )
    res_waste_index = MRTA_CFG_Model.cal_res_waste_index(
        uav_manager, task_manager, coalition_manager, hyper_params
    )
    dist_cost_index = MRTA_CFG_Model.cal_dist_cost_index(
        uav_manager, task_manager, coalition_manager, hyper_params, model_hyper_params=model_hparams
    )
    threat_cost_index = MRTA_CFG_Model.cal_threat_cost_index(
        uav_manager, task_manager, coalition_manager, hyper_params, model_hyper_params=model_hparams
    )
    print("task_comp_index", task_comp_index)
    print("res_sat_index", res_sat_index)
    print("res_waste_index", res_waste_index)
    print("dist_cost_index", dist_cost_index)
    print("threat_cost_index", threat_cost_index)


def test_model_prefer():
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

    t = task_list[0]
    model_hparams = MRTA_CFG_Model_HyperParams()
    coalition_manager = CoalitionManager(uav_manager.get_ids(), task_manager.get_ids())
    for uav in uav_list:
        coalition_manager.assign(uav.id, t.id)

    # test
    u = uav_list[0]
    uav_benefit = MRTA_CFG_Model.cal_uav_benefit(u, t, uav_list, resources_num=resources_num)
    # print("uav_benefit", uav_benefit)
    tp = t
    tq = task_manager.get_free_uav_task(resources_num)

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


def test():
    # check Activations
    # Activations.plot_all()
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

    # task_manager.format_print()
    # uav_manager.format_print()

    t = task_list[0]
    coalition_manager = CoalitionManager(uav_manager.get_ids(), task_manager.get_ids())
    for uav in uav_list:
        coalition_manager.assign(uav.id, t.id)
    # coalition_manager.format_print()

    # eval_cs = MRTA_CFG_Model.cal_cs_eval(
    #     uav_manager,
    #     task_manager,
    #     coalition_manager,
    #     hyper_params,
    # )
    # print("eval_cs", eval_cs)

    # cs_eval_beta = MRTA_CFG_Model.cal_cs_eval_beta(
    #     uav_manager,
    #     task_manager,
    #     coalition_manager,
    #     hyper_params,
    # )
    # print("cs_eval_beta", cs_eval_beta)

    # payoff_dict: np.ndarray = MRTA_CFG_Model.cal_coalition_payoff(
    #     t, uav_list, resources_num, method="monte_carlo"
    # )
    # payoff_sum = sum(payoff_dict.values())
    # print("payoff_vector", payoff_dict)
    # print("payoff_sum", payoff_sum)

    model_hparams = MRTA_CFG_Model_HyperParams()
    coalition_eval = MRTA_CFG_Model.cal_coalition_eval(t, uav_list, resources_num, model_hparams)
    print("coalition_eval", coalition_eval)


if __name__ == "__main__":
    test()
