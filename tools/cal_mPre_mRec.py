import numpy as np
import os
from scipy import stats

NUM_CLASSES = 13

at = 0.5
tp = np.zeros(13)
fp = np.zeros(13)
total_gt_ins = np.zeros(NUM_CLASSES)
total_pred_ins = np.zeros(NUM_CLASSES)

for area_id in [5]:
    file_path = "/wdc/ncb-old/SoftGroup_multi_versions_manage/SoftGroup/results_for_6fold/log" + str(area_id) + "_test"
    scan_ids = os.listdir(file_path + '/gt_instance/')
    for i, scan_id in enumerate(scan_ids):
        # load pred_masks by semantic class
        print("room:{}".format(i))
        pred_datas = np.loadtxt(file_path + '/pred_instance/' + scan_id, str)
        gt_ins = np.loadtxt(file_path + '/gt_instance/' + scan_id, np.int_)
        gt_sem = np.load(file_path + '/semantic_label/' + scan_id.rstrip('.txt') + '.npy')
        un = np.unique(gt_ins)
        un = np.delete(un, 0)
        gt_ins_num = un.shape[0]

        pts_in_pred = [[] for itmp in range(NUM_CLASSES)]
        for pred_data in pred_datas:
            if float(pred_data[2]) < 0.09:
                continue
            pred_mask = np.loadtxt(file_path + '/pred_instance/' + pred_data[0])
            tmp = (pred_mask == 1)
            sem_seg_i = int(pred_data[1]) - 1
            pts_in_pred[sem_seg_i] += [tmp]

        un = np.unique(gt_ins)
        pts_in_gt = [[] for itmp in range(NUM_CLASSES)]
        for ig, g in enumerate(un):
            if g < 1000:
                continue
            tmp = (gt_ins == g)
            sem_seg_i = int(stats.mode(gt_sem[tmp])[0])
            pts_in_gt[sem_seg_i] += [tmp]

        for i_sem in range(NUM_CLASSES):
            total_gt_ins[i_sem] += len(pts_in_gt[i_sem])
            total_pred_ins[i_sem] += len(pts_in_pred[i_sem])
            for pred_ins in pts_in_pred[i_sem]:
                ovmax = -1
                for gt_ins in pts_in_gt[i_sem]:
                    if len(pred_ins) == 0:
                        continue
                    intersect = (np.array(pred_ins).astype(np.int_) | np.array(gt_ins).astype(np.int_))
                    union = (np.array(pred_ins).astype(np.int_) & np.array(gt_ins).astype(np.int_))
                    if np.sum(union) == 0:
                        continue
                    iou = float(np.sum(intersect)) / np.sum(union)
                    if iou > ovmax:
                        ovmax = iou
                        igmax = ig

                if ovmax >= at:
                    tp[i_sem] += 1  # true
                else:
                    fp[i_sem] += 1  # false positive

print("tp:{}".format(tp))
print("fp:{}".format(fp))
print("total_gt_ins:{}".format(total_gt_ins))
print("total_pred_ins:{}".format(total_pred_ins))

mPrec = np.sum(np.array(tp) / (np.array(tp) + np.array(fp) + np.ones(NUM_CLASSES) * 1e-6)) / 13
mRec = np.sum(np.array(tp) / total_gt_ins) / 13
print("mPrec:{}".format(mPrec))
print("mRec:{}".format(mRec))

