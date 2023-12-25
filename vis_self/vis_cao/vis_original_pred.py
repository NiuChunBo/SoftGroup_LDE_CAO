import open3d as o3d
import sys
import argparse
import numpy as np


def process_file(scan_path):
    pcd = o3d.io.read_point_cloud('../original_results/instance_pred/' + scan_path.rstrip('_instance_gt.ply') + '_instance_pred.ply')

    # remove points which sem = 0
    sem = np.load('../../results_for_6fold/log5_test/semantic_label/' + scan_path.rstrip('_instance_gt.ply') + '.npy')
    index = np.where(sem == 6)
    # ind = np.where(sem == 7)
    # ind2 = np.where(sem == 8)
    # index = tuple(np.concatenate((ind, ind2), 1))
    pcd.points = o3d.cuda.pybind.utility.Vector3dVector(np.asarray(pcd.points)[index])
    pcd.colors = o3d.cuda.pybind.utility.Vector3dVector(np.asarray(pcd.colors)[index])

    o3d.visualization.draw_geometries([pcd], 'instance_original_pred of ' + scan_path.rstrip('_instance_gt.ply'), 1920, 1080, 0, 0)


if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 添加 "file_path" 参数
    parser.add_argument("file_path", help="The path to the file")

    # 解析命令行参数
    args = parser.parse_args()

    # 使用 "file_path" 参数
    print("File path:", '../original_results/instance_pred/' + args.file_path.rstrip('_instance_gt.ply') + '_instance_pred.ply')
    # 获取命令行参数
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        process_file(file_path)
    else:
        print("请提供文件路径作为命令行参数")
