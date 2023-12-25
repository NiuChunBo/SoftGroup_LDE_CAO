import os
import numpy as np
import subprocess

# files_name = os.listdir('./results_for_area5/gt_instance/')
files_name = os.listdir('../CAO_ablation1_results/gt_instance/')
files_name = [file_name.rstrip('.txt') for file_name in files_name]
tasks = ['input', 'semantic_pred', 'semantic_gt', 'instance_gt', 'instance_pred', 'offset_semantic_pred']

i = 0
for room_name in files_name:
    print(f"room：{i}/{len(files_name)}", end="\r")
    i += 1

    for task in tasks:
        subprocess.run(['python ../../tools/visualization_bak.py --prediction_path ../CAO_ablation1_results/ --task '+ task
                        +' --out ./vis_self/CAO_ablation1_saved_results/'+ task + '/' + room_name + '_' + task +'.ply --room_name '+ room_name], shell=True)
        print('saving room: {} task: {}'.format(room_name, task))

# 检查是否全部保存
for task in tasks:
    save_files_name = os.listdir('../CAO_ablation1_saved_results/' + task + '/')
    if len(save_files_name) == len(files_name):
        print("task: {} {}/{}个场景全部成功保存！".format(task, len(save_files_name), len(files_name)))
    else:
        print("task: {} {}/{} 有场景未成功保存！".format(task, len(save_files_name), len(files_name)))
