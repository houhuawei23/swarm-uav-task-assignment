# Repotitory for Swarm UAV Task Assignment

## TODO

- [x] implement the algorithm in the [csci2024@薛舒心](https://doi.org/10.1360/ssi-2024-0167), `./solvers/csci.py`
  - [x] add time constraints
  - [ ] uavs and tasks cluster
- [x] fix bugs in CoalitionFormationGame task assignment algorithm.
- [x] Calculate various evaluation indicators, such as:
  - task completion rate, and resource use rate
- [x] Implement algorithm in [iros2024@LiwangZhang](https://doi.org/10.1109/IROS58592.2024.10801429), `./solvers/iros.py`
- [x] Implement algorithm in [icra2024@LiwangZhang](https://doi.org/10.1109/ICRA57147.2024.10611476), `./solvers/icra.py`
- Warning: iros2024@LiwangZhang and icra2024@LiwangZhang both need random sample in trigger uav stage! Otherwise, the alg may be vibrate and stuck in a deadlock.

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
def run_solver(
    uav_manager: UAVManager,
    task_manager: TaskManager,
    hyper_params: HyperParams,
    cmd_args: CmdArgs,
    result_queue: Queue = None,
):
    coalition_manager = CoalitionManager(uav_manager.get_ids(), task_manager.get_ids())
    if cmd_args.choice == "csci":
        solver = ChinaScience2024_CoalitionFormationGame(
            uav_manager, task_manager, coalition_manager, hyper_params
        )
    elif cmd_args.choice == "iros":
        solver = IROS2024_CoalitionFormationGame(
            uav_manager, task_manager, coalition_manager, hyper_params
        )
    elif cmd_args.choice == "icra":
        solver = ICRA2024_CoalitionFormationGame(
            uav_manager, task_manager, coalition_manager, hyper_params
        )
    elif cmd_args.choice == "enum":
        solver = EnumerationAlgorithm(uav_manager, task_manager, coalition_manager, hyper_params)
    else:
        raise ValueError("Invalid choice")

    start_time = time.time()
    solver.run_allocate(debug=False)
    end_time = time.time()
    elapsed_time = end_time - start_time

    if result_queue is not None:
        result_queue.put(elapsed_time)

    print(f"{cmd_args.choice} Result: {coalition_manager}")

    eval_reuslt = evaluate_assignment(
        uav_manager, task_manager, coalition_manager.task2coalition, hyper_params.resources_num
    )
    print(f"Eval Result: {eval_reuslt}")
    coalition_manager.plot_map(
        uav_manager, task_manager, hyper_params, cmd_args.output_path, plot_unassigned=True
    )

```

```bash
$ python ./sim.py --test_case ./tests/case0.json --choice icra

Using test case: ./tests/case0.json, output path: ./.images/.result.png, choice: icra
Using UAV Type: <class 'solvers.icra2024.AutoUAV'>
HyperParams(resources_num=2, map_shape=(np.int64(21), np.int64(21), 0), alpha=1.0, beta=10.0, gamma=0.05, mu=-1.0, max_iter=25)
UAVManager with 3 uavs.
  AutoUAV(id=1, position=Point(xyz=array([0, 0, 0])), resources=array([5, 3]), value=10, max_speed=20, mass=1.0, fly_energy_per_time=1.1958796458778673, hover_energy_per_time=2.038716644186131, uav_manager=None, task_manager=None, coalition_manager=None, hyper_params=None, uav_update_step_dict={}, changed=False)
  ...
TaskManager with 2 tasks.
  Task(id=1, position=Point(xyz=array([5, 5, 0])), required_resources=array([4, 2]), time_window=[0, 100], threat=0.5, execution_time=2.821083273573082, resources_nums=2)
  ...
...
icra Result: {1: [1], 2: [3], None: [2]}
Eval Result: EvaluationResult(completion_rate=0.5, resource_use_rate=0.7333333333333334)
```

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
