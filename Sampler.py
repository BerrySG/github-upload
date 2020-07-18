
# Sorry I've only finished this part, the rest runs super slow... Never seen any result coming
import numpy as np
import cv2


class PatchSampler():
    def __init__(self, train_images_list, gt_segmentation_maps_list, classes_colors, patchsize):

        self.train_images_list = train_images_list
        self.gt_segmentation_maps_list = gt_segmentation_maps_list
        self.class_colors = classes_colors
        self.patchsize = patchsize

    # Function for sampling patches for each class
    # provide your implementation
    # should return extracted patches with labels
    def extractpatches(self):
        img_num = len(self.train_images_list)
        all_possible = [[] for _ in range(4)]
        for n in range(img_num):
            img = cv2.imread('images/' + self.train_images_list[n])
            gt = cv2.imread('images/' + self.gt_segmentation_maps_list[n])
            resolution = img.shape
            for i in range(0, resolution[0]-self.patchsize, 2):
                for j in range(0, resolution[1]-self.patchsize, 2):
                    img_patch = img[i:i + self.patchsize, j:j + self.patchsize, :]
                    gtseg = gt[i:i + self.patchsize, j:j + self.patchsize, :]
                    label = self.generate_label(gtseg[:, :, 0].flatten())
                    all_possible[label].append(img_patch)
        samples = [[] for _ in range(4)]
        res = []
        for i in range(self.class_colors):
            np.random.shuffle(all_possible[i])
            samples[i] = all_possible[i][0:4000].copy()
            for j in range(len(samples[i])):
                sample = [samples[i][j], i]
                res.append(sample)
        np.random.shuffle(res)
        temp = np.array(res)
        patch_list = temp[:, 0].copy()
        gtseg_list = temp[:, 1].copy()
        return patch_list.tolist(), gtseg_list.tolist()

    def generate_label(self, gtseg):
        return np.argmax(np.bincount(gtseg))
