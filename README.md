# Repotitory for Swarm UAV Task Assignment

- try to implement the algorithm in the [paper](https://doi.org/10.1360/ssi-2024-0167).

## Show Cases

下图左侧为分配前示意图，右侧为联盟博弈分配得到的结果（简单样例 `case0.json`）。

<p align="center"> 
<img src="./assets/init.png" width=45%/> 
<img src="./assets/assigned.png" width=45%/> 
</p>

下图左侧为暴力搜索所得的最佳分配结果，右侧为联盟博弈得到的结果（`case1.json`）。

<p align="center"> 
<img src="./assets/case1_enumeration_result.png" width=45%/> 
<img src="./assets/case1_coalition_game_result.png" width=45%/> 
</p>

`case3.json`(10 uavs, 5 tasks), 下图为联盟博弈分配得到的结果，暴力搜索没有在规定时间内得到结果。

<p align="center"> 
<img src="./assets/case3_coalition_game_result.png" width=75%/> 
</p>

```python
def run_enumeration(
    uav_manager: UAVManager,
    task_manager: TaskManager,
    hyper_params: HyperParams,
    result_queue: Queue = None,  # for return result
):
    print("Enumeration")
    enumeration_algorithm = EnumerationAlgorithm(uav_manager, task_manager, hyper_params)
    start_time = time.time()
    best_assignment, best_score = enumeration_algorithm.solve()
    end_time = time.time()
    elapsed_time = end_time - start_time
    if result_queue is not None:
        result_queue.put(elapsed_time)
    print(f"Elapsed Time: {elapsed_time}")
    print(f"Best Assignment: {best_assignment}")
    print(f"Best Score: {best_score}")

    enu_coalition_set = CoalitionManager(uav_manager, task_manager, assignment=best_assignment)
    enu_coalition_set.plot_map(".enumeration_result.png")


def run_coalition_game(
    uav_manager: UAVManager,
    task_manager: TaskManager,
    hyper_params: HyperParams,
    result_queue: Queue = None,
):
    print("Coalition Game")
    coalition_manager = CoalitionManager(uav_manager, task_manager)
    game = CoalitionFormationGame(uav_manager, task_manager, coalition_manager, hyper_params=hyper_params)

    # coalition_set.plot_map()
    start_time = time.time()
    game.run(debug=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    if result_queue is not None:
        result_queue.put(elapsed_time)
    print(f"Coalition Game Result: {coalition_manager}")

    eval_reuslt = evaluate_assignment(
        uav_manager, task_manager, coalition_manager.task2coalition, hyper_params.resources_num
    )
    print(f"Eval Result: {eval_reuslt}")
    coalition_manager.plot_map(".coalition_game_result.png", plot_unassigned=True)


```

```bash
$ python sim.py --test_case ../tests/case1.json

Using test case: ../tests/case1.json
UAVManager managing 5 UAVs
  u1, re=[5 3], pos=[0 0 0], val=10, ms=20
  u2, re=[ 7 11], pos=[2.5 3.5 0. ], val=10, ms=20
  u3, re=[2 7], pos=[10 10  0], val=15, ms=25
  u4, re=[19  4], pos=[16.  17.5  0. ], val=15, ms=25
  u5, re=[ 6 21], pos=[20 20  0], val=20, ms=30
TaskManager with 2 tasks.
  T1: re=[10 12], pos=[5 5 0], tw=[0, 100], thr=0.5
  T2: re=[25  8], pos=[15 15  0], tw=[0, 100], thr=0.7
---
Coalition Game
Iteration 0 begin.
Cur coalition set: {1: [], 2: [], None: [1, 2, 3, 4, 5]}
check_stability True, Iteration 1 end.
Coalition Game Result: {1: [2], 2: [4], None: [1, 3, 5]}
---
Enumeration
All 243 assignments:
Best Assignment: {1: [1, 2], 2: [4, 5]}
Best Score: 54.851725195536254
```

## TODO

- [x] implement the algorithm in the [paper](https://doi.org/10.1360/ssi-2024-0167).
  - [x] add time constraints
  - [ ] uavs and tasks cluster
- [x] fix bugs in CoalitionFormationGame task assignment algorithm.
- [x] Calculate various evaluation indicators, such as:
  - task completion rate, and resource use rate

## Project Structure

- `src/`:
  - `base.py`: base class for the project.
  - `uav.py`: the class for `UAV` and `UAVManager`.
  - `task.py`: the class for `Task` and `TaskManager`.
  - `coalition.py`: the class for `CoalitionSet`.
  - `utils.py`: the utility functions.
  - `task_assign.py`: implement `EnumerationAlgorithm`.
  - `game.py`: implement `CoalitionFormationGame`.
  - `sim.py`: the main simulation script.
  - `gen.py`: generate test data, in json format.

- `tests/`
  - case0.json: 3 uavs, 2 tasks; no coalition.
  - case1.json: 5 uavs, 2 tasks.
  - case2.json: 5 uavs, 3 tasks.
  - case3.json: 10 uavs, 5 tasks.
  - case4.json: 50 uavs, 5 tasks.
