from RandomForest import Forest
from Sampler import PatchSampler
import numpy as np
import cv2
from Tree import DecisionTree


def main():
    file = open('images/train_images.txt', 'r')
    train_spc = file.readlines()
    file.close()
    info_num = train_spc[0].split()
    pic_num = np.int(info_num[0])
    class_num = np.int(info_num[1])
    train_list = []
    gt_list = []
    for i in range(1, pic_num+1):
        spec = train_spc[i].split()
        train_list.append(spec[0])
        gt_list.append(spec[1])
    samples = PatchSampler(train_list, gt_list, class_num, 16)
    train_patch, gtseg_patch = samples.extractpatches()
    tree_param = {'depth': 15, 'pixel_locations': 100, 'random_color_values': 10, 'no_of_thresholds': 50,
                  'minimum_patches_at_leaf': 20, 'classes': 4}
    fir_tree = DecisionTree(train_patch, gtseg_patch, tree_param)
    fir_tree.train()


main()

