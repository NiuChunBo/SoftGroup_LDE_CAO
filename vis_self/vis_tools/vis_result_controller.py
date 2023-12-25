import tkinter as tk
import os
import subprocess
import open3d as o3d
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor


# 定义第一个程序的执行函数
def run_program1():
    global scan_id, scan_num, instance_gt_files_path, scan_files
    subprocess.call(['python', './vis_pred_self.py', scan_files[scan_id]])
    # subprocess.call(['python', './vis_gt.py'])


# 定义第二个程序的执行函数
def run_program2():
    subprocess.call(['python', './vis_pred_original.py', scan_files[scan_id]])
    # subprocess.call(['python', './vis_gt.py'])
    
   
# 定义第三个程序的执行函数
def run_program3():
    subprocess.call(['python', './vis_input.py', scan_files[scan_id]])
    # subprocess.call(['python', './vis_input.py'])
    
    
# 定义第四个程序的执行函数
def run_program4():
    subprocess.call(['python', './vis_gt.py', scan_files[scan_id]])
    # subprocess.call(['python', './vis_gt.py'])
    

# 定义第五个程序的执行函数
def run_program5():
    subprocess.call(['python', './vis_sem_original.py', scan_files[scan_id]])
    # subprocess.call(['python', './vis_sem_original.py'])
    
    
# 定义第六个程序的执行函数
def run_program6():
    subprocess.call(['python', './vis_sem_self.py', scan_files[scan_id]])
    # subprocess.call(['python', './vis_sem_self.py'])
    

# 定义第七个程序的执行函数
def run_program7():
    subprocess.call(['python', './vis_sem_label.py', scan_files[scan_id]])
    # subprocess.call(['python', './vis_sem_label.py'])


# 打开上一个场景
def prev_button_clicked():
    global scan_id, scan_num, instance_gt_files_path, scan_files
    if scan_id <= 0:
        print("scan_id已经为0")
    else:
        scan_id -= 1
        print(scan_id)
        text_label.config(text=scan_files[scan_id].rstrip('_instance_gt.ply') + '\n' + str((scan_id + 1)) + '/' + str(scan_num))


# 打开下一个场景
def next_button_clicked():
    global scan_id, scan_num, instance_gt_files_path, scan_files
    if scan_id == scan_num - 1:
        print("scan_id已经最大")
    else:
        scan_id += 1
        print(scan_id)
        text_label.config(text=scan_files[scan_id].rstrip('_instance_gt.ply') + '\n' + str((scan_id + 1)) + '/' + str(scan_num))
        
        
# 显示场景的七种状态
def show_button_clicked():
    global scan_id, scan_num, instance_gt_files_path, scan_files
    if scan_id < 0 or scan_id >67:
        print("scan_id {} 不在正常范围！".format(scan_id))
    else:
        # 创建线程池
        executor = ThreadPoolExecutor(max_workers=7)
        # 提交任务到线程池
        executor.submit(run_program1)
        executor.submit(run_program2)
        
        executor.submit(run_program3)
        executor.submit(run_program4)
        
        # executor.submit(run_program5)
        # executor.submit(run_program6)
        # executor.submit(run_program7)
        # 关闭线程池
        executor.shutdown()


if __name__ == '__main__':
    # 指定场景文件的根目录
    instance_gt_files_path = f"../cao_ablation_results/instance_gt"
    # 获取场景的文件列表
    scan_files = os.listdir(instance_gt_files_path)
    scan_num = len(scan_files)
    scan_id = -1

    print([scan_file.rstrip('_instance_gt.ply') for scan_file in scan_files])
    # 创建窗口
    window = tk.Tk()

    # 设置窗口大小
    window.geometry("350x300")

    # 创建文本显示框
    text_label = tk.Label(window, text="Area_5_scan_name_id\n0/68", font=("Arial", 16))
    text_label.pack(pady=20)

    # 创建上一个按钮
    prev_button = tk.Button(window, text="上一个", font=("Arial", 20), command=prev_button_clicked)
    prev_button.pack(pady=10)

    # 创建下一个按钮
    next_button = tk.Button(window, text="下一个", font=("Arial", 20), command=next_button_clicked)
    next_button.pack(pady=10)
    
    # 创建显示按钮
    next_button = tk.Button(window, text="show scan", font=("Arial", 20), command=show_button_clicked)
    next_button.pack(pady=10)

    # 进入主循环
    window.mainloop()
