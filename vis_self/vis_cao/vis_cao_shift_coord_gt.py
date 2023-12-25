import open3d as o3d
import sys
import argparse
import numpy as np


def process_file(scan_path):
    pcd = o3d.io.read_point_cloud('../cao_ablation_results/instance_pred/' + scan_path.rstrip('_instance_gt.ply') + '_instance_pred.ply')

    # remove points which sem = 0
    sem = np.load('../ablation_results/cao_test_results/semantic_label/' + scan_path.rstrip('_instance_gt.ply') + '.npy')
    offset_pred = np.load('../ablation_results/cao_test_results/offset_pred/' + scan_path.rstrip('_instance_gt.ply') + '.npy')
    ind = np.where(sem == 7)
    # ind = np.where(sem == 7)
    # ind2 = np.where(sem == 8)
    # index = tuple(np.concatenate((ind, ind2), 1))
    a = [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.1, 1.2, 0.5, 0.7, 1.0, 1.0, 0.6]
    points = np.asarray(pcd.points)
    for i in range(13):
        index = np.where(sem == i)
        points[index] += a[i] * np.asarray(offset_pred)[index]
    pcd.points = o3d.cuda.pybind.utility.Vector3dVector(points[ind])
    pcd.colors = o3d.cuda.pybind.utility.Vector3dVector(np.asarray(pcd.colors)[ind])

    o3d.visualization.draw_geometries([pcd], '类感知偏移后坐标 of ' + scan_path.rstrip('_instance_gt.ply'), 1920, 1080, 0, 0)


if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 添加 "file_path" 参数
    parser.add_argument("file_path", help="The path to the file")

    # 解析命令行参数
    args = parser.parse_args()

    # 使用 "file_path" 参数
    print("File path:", '../cao_ablation_results/instance_pred/' + args.file_path.rstrip('_instance_gt.ply') + '_instance_pred.ply')
    # 获取命令行参数
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        process_file(file_path)
    else:
        print("请提供文件路径作为命令行参数")
