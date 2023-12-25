import numpy as np

tp1 = np.array([43, 44, 188, 45, 46, 29, 82, 135, 50, 43, 6, 23, 451])
fp1 = np.array([1, 0, 54, 5, 8, 2, 3, 24, 25, 15, 3, 3, 128])
gt_ins1 = np.array([55, 44, 234, 61, 57, 29, 86, 155, 69, 90, 6, 27, 777])

tp2 = np.array([37, 40, 173, 4, 13, 8, 72, 60, 25, 17, 5, 6, 218])
fp2 = np.array([3, 0, 75, 27, 18, 2, 29, 49, 17, 26, 5, 7, 180])
gt_ins2 = np.array([81, 50, 283, 11, 19, 8, 93, 545, 46, 47, 6, 17, 505])

tp3 = np.array([23, 23, 80, 11, 9, 7, 35, 57, 28, 25, 7, 10, 188])
fp3 = np.array([0, 0, 20, 5, 15, 1, 3, 0, 6, 9, 1, 3, 53])
gt_ins3 = np.array([37, 23, 159, 13, 12, 8, 37, 67, 30, 41, 9, 12, 335])

tp4 = np.array([46, 49, 190, 3, 28, 21, 90, 124, 52, 51, 9, 8, 316])
fp4 = np.array([3, 0, 61, 2, 27, 9, 26, 9, 26, 30, 3, 6, 110])
gt_ins4 = np.array([73, 50, 280, 3, 38, 40, 107, 159, 78, 98, 14, 10, 671])

tp5 = np.array([68, 68, 259, 0, 43, 41, 112, 228, 92, 126, 9, 35, 389])
fp5 = np.array([0, 0, 108, 4, 23, 3, 39, 9, 38, 76, 0, 5, 209])
gt_ins5 = np.array([76, 68, 342, 3, 74, 52, 127, 258, 154, 216, 11, 42, 918])

tp6 = np.array([46, 48, 202, 52, 40, 29, 90, 163, 59, 49, 4, 29, 434])
fp6 = np.array([2, 0, 41, 7, 11, 3, 11, 12, 10, 26, 3, 4, 122])
gt_ins6 = np.array([63, 49, 247, 68, 54, 31, 93, 179, 77, 90, 9, 29, 685])

tp3_2 = np.array([23, 23, 88, 11, 9., 7., 35., 60, 29., 25., 7., 10., 217.])
fp3_2 = np.array([0., 0., 39., 8., 30., 1., 11., 6., 10., 18., 1., 4., 139.])

tp = tp1 + tp2 + tp3_2 + tp4 + tp5 + tp6
fp = fp1 + fp2 + fp3_2 + fp4 + fp5 + fp6
gt_ins = gt_ins1 + gt_ins2 + gt_ins3 + gt_ins4 + gt_ins5 + gt_ins6

prec = np.sum(tp / (tp + fp)) / 13
rec = np.sum(tp / gt_ins) / 13

print("mPrec:{}".format(prec))
print("mRec:{}".format(rec))