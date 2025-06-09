#! /bin/bash
# run test

# run plot
result_dir="./.results"
target_dir="./.results"
## results_uav_num_all_0518
python ./main.py plot -f $result_dir/results_uav_num_all_0518.yaml -x uav_num --labels all \
--choices CSCI2024_Xue IROS2024_LiwangZhang ICRA2024_LiwangZhang Centralized Distributed \
--save_dir $target_dir/uav_num_all_0518/
## results_task_num_0518
python ./main.py plot -f $result_dir/results_task_num_0518.yaml -x task_num --labels all \
--save_dir $target_dir/task_num_all_0518/ 

### resource contribution weight
python ./main.py plot -f $result_dir/results_hyper_params.resource_contribution_weight_all_0519.yaml -x hyper_params.resource_contribution_weight --labels all \
--choices CSCI2024_Xue IROS2024_LiwangZhang ICRA2024_LiwangZhang Centralized Distributed  \
--save_dir $target_dir/results_hyper_params.resource_contribution_weight_alzl_0519

### path cost weight
python ./main.py plot -f $result_dir/results_hyper_params.path_cost_weight_all_0519.yaml -x hyper_params.path_cost_weight --labels all \
--save_dir $target_dir/results_hyper_params.path_cost_weight_all_0519

### threat loss weight
python ./main.py plot -f $result_dir/results_hyper_params.threat_loss_weight_all_0519.yaml -x hyper_params.threat_loss_weight --labels all \
--save_dir $target_dir/results_hyper_params.threat_loss_weight_all_0519

### resource waste weight
python ./main.py plot -f $result_dir/results_hyper_params.resource_waste_weight_all_0519.yaml -x hyper_params.resource_waste_weight --labels all \
--save_dir $target_dir/results_hyper_params.resource_waste_weight_all_0519
## Ablation Study
### uav num
python ./main.py plot -f $result_dir/results_uav_num_ablation_0519.yaml -x uav_num --labels all \
--save_dir $target_dir/uav_num_all_ablation_0519/ 
### task num
python ./main.py plot -f $result_dir/results_task_num_ablation_0519.yaml -x task_num --labels all \
--save_dir $target_dir/task_num_all_ablation_0519/ 
### preference
python ./main.py plot -f $result_dir/results_uav_num_preference_0519.yaml -x uav_num --labels all \
--save_dir $target_dir/uav_num_all_preference_0519/
