import os
import numpy as np
from scipy import stats

NUM_CLASSES = 13

pred_data_label_filenames = []
gt_instance_paths = []
semantic_pred_paths = []
semantic_label_paths = []
pred_label_paths = []

# for i in [1, 2, 3, 4, 5, 6]:
for i in [5]:
    print("Area", i)
    file_name = '/wdc/ncb-old/SoftGroup_multi_versions_manage/SoftGroup/results_for_6fold/log{' \
                '}_test/'.format(i)

    scan_ids = [file.rstrip('.txt') for file in os.listdir(file_name + 'gt_instance/')]
    gt_instance_paths += [file_name + 'gt_instance/' + file + '.txt' for file in scan_ids]
    semantic_pred_paths += [file_name + 'semantic_pred/' + file + '.npy' for file in scan_ids]
    semantic_label_paths += [file_name + 'semantic_label/' + file + '.npy' for file in scan_ids]
    pred_label_paths += [file_name + 'ins_label_from_visualization/' + file + '.npy' for file in scan_ids]

num_room = len(gt_instance_paths)

# Initialize...
# precision & recall
total_gt_ins = np.zeros(NUM_CLASSES)
total_pred_ins = np.zeros(NUM_CLASSES)
at = 0.5
tp_self = [0] * NUM_CLASSES
fp_self = [0] * NUM_CLASSES
fn_self = [0] * NUM_CLASSES

tp_self_2 = [0] * NUM_CLASSES
fp_self_2 = [0] * NUM_CLASSES

total_tp_fp = [0] * NUM_CLASSES

for i in range(num_room):
    print(f"room：{i}/{num_room}", end="\r")
    print("room:{}".format(i))

    gt_ins = np.loadtxt(gt_instance_paths[i], np.int_)
    pred_sem = np.load(semantic_pred_paths[i])
    gt_sem = np.load(semantic_label_paths[i])
    pred_ins = np.load(pred_label_paths[i])

    # instance
    un = np.unique(pred_ins)
    pts_in_pred = [[] for itmp in range(NUM_CLASSES)]
    for ig, g in enumerate(un):  # each object in prediction
        if g == -100:
            continue
        tmp = (pred_ins == g)
        sem_seg_i = int(stats.mode(pred_sem[tmp])[0])
        pts_in_pred[sem_seg_i] += [tmp]

    un = np.unique(gt_ins)
    pts_in_gt = [[] for itmp in range(NUM_CLASSES)]
    for ig, g in enumerate(un):
        if g < 1000:
            continue
        tmp = (gt_ins == g)
        sem_seg_i = int(stats.mode(gt_sem[tmp])[0])
        pts_in_gt[sem_seg_i] += [tmp]

    # instance precision & recall
    for i_sem in range(NUM_CLASSES):
        gtflag = np.zeros(len(pts_in_gt[i_sem]))
        total_gt_ins[i_sem] += len(pts_in_gt[i_sem])
        total_pred_ins[i_sem] += len(pts_in_pred[i_sem])

        for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
            ovmax = -1.
            for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                union = (ins_pred | ins_gt)
                intersect = (ins_pred & ins_gt)
                iou = float(np.sum(intersect)) / np.sum(union)

                if iou > ovmax:
                    ovmax = iou
                    igmax = ig

            if ovmax >= at:
                tp_self[i_sem] += 1  # true
            else:
                fp_self[i_sem] += 1  # false positive


print("tp_self:{}".format(tp_self))
print("fp_self:{}".format(fp_self))

print("total_gt_ins:{}".format(total_gt_ins))

mPrec = np.sum(np.array(tp_self) / (np.array(tp_self) + np.array(fp_self) + np.ones(NUM_CLASSES) * 1e-6)) / 13
mRec = np.sum(np.array(tp_self) / total_gt_ins) / 13
print("mPrec:{}".format(mPrec))
print("mRec:{}".format(mRec))

