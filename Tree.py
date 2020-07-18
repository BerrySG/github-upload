import numpy as np
from Node import Node


class DecisionTree():
    def __init__(self, patches, labels, tree_param):

        self.patches, self.labels = patches, labels
        self.depth = tree_param['depth']
        self.pixel_locations = tree_param['pixel_locations']
        self.random_color_values = tree_param['random_color_values']
        self.no_of_thresholds = tree_param['no_of_thresholds']
        self.minimum_patches_at_leaf = tree_param['minimum_patches_at_leaf']
        self.classes = tree_param['classes']
        self.nodes = []

    # Function to train the tree
    # provide your implementation
    # should return a trained tree with provided tree param
    def train(self):
        root_node = Node()
        self.grow_tree(root_node, self.patches, self.labels)
        print('training completed')

    # Function to predict probabilities for single image
    # provide your implementation
    # should return predicted class for every pixel in the test image
    def predict(self, I):
        pass

    # Function to get feature response for a random color and pixel location
    # provide your implementation
    # should return feature response for all input patches
    def getFeatureResponse(self, patches, feature):
        color, cor, _ = feature
        resp = patches[cor[0]][cor[1]][color]
        return resp

    # Function to get left/right split given feature responses and a threshold
    # provide your implementation
    # should return left/right split
    def getsplit(self, responses, threshold):
        if responses <= threshold:
            return 0
        else:
            return 1

    # Function to get a random pixel location
    # provide your implementation
    # should return a random location inside the patch
    def generate_random_pixel_location(self):
        random_pixeloc = []
        for i in range(self.pixel_locations):
            x_cor = np.random.randint(0, 16)
            y_cor = np.random.randint(0, 16)
            random_pixeloc.append([x_cor, y_cor])
        return random_pixeloc

    # Function to compute entropy over incoming class labels
    # provide your implementation
    def compute_entropy(self, labels):
        entropy = 0
        total_num = len(labels)
        for i in range(4):
            labels.count(i)
            prob = labels.count(i) / total_num
            if prob != 0:
                entropy += prob + np.log2(prob)
        return entropy * -1

    # Function to measure information gain for a given split
    # provide your implementation
    def get_information_gain(self, Entropyleft, Entropyright, EntropyAll, Nall, Nleft, Nright):
        gain = EntropyAll - ((Nleft/Nall) * Entropyleft + (Nright/Nall) * Entropyright)
        return gain

    # Function to get the best split for given patches with labels
    # provide your implementation
    # should return left,right split, color, pixel location and threshold
    def best_split(self, patches, labels):
        entropy_all = self.compute_entropy(labels)
        entropy_final_left = 0
        entropy_final_right = 0
        max_gain = -np.inf
        right_num = 0
        left_num = 0
        patch_size = len(patches)
        left_labels = []
        right_labels = []
        # Get binary test set
        bin_test = []
        for i in range(self.random_color_values):
            channel = np.random.randint(3)
            ran_pixel = self.generate_random_pixel_location()
            for j in range(len(ran_pixel)):
                for k in range(self.no_of_thresholds):
                    threshold = np.random.randint(0, 256)
                    bin_test.append([channel, ran_pixel[j], threshold])

        for i in range(len(bin_test)):
            sizeleft = 0
            sizeright = 0

            for j in range(len(patches)):
                fea_resp = self.getFeatureResponse(patches[j], bin_test[i])
                node_split = self.getsplit(fea_resp, bin_test[i][2])
                if node_split == 0:
                    left_labels.append(labels[j])
                    sizeleft += 1
                else:
                    right_labels.append(labels[j])
                    sizeright += 1
            entropy_left = self.compute_entropy(left_labels)
            entropy_right = self.compute_entropy(right_labels)
            gain = self.get_information_gain(entropy_left, entropy_right, entropy_all, patch_size, sizeleft, sizeright)
            if gain > max_gain:
                split = bin_test[i]
                max_gain = gain
                left_num = sizeleft
                right_num = sizeright
                entropy_final_left = entropy_left
                entropy_final_right = entropy_right
        if max_gain == 0 or right_num == 0 or left_num == 0:
            return False, False, False
        dataLeft = []
        dataRight = []
        for i in range(patch_size):
            resp = self.getFeatureResponse(patches[i], split)
            if resp < split[2]:
                dataLeft.append(patches[i].copy())
            else:
                dataRight.append(patches[i].copy())
        return dataLeft, dataRight, split

    def grow_tree(self, node, patches, labels):
        self.nodes.append(node)
        patch_size = len(patches)
        if patch_size < self.minimum_patches_at_leaf or node.depth > self.depth:
            node.create_leafNode(patches, labels)
            return True
        else:
            dataLeft, dataRight, split = self.best_split(patches, labels)
            if dataLeft != False:
                node.type = 'split'
                total_num = len(labels)
                for i in range(4):
                    print('class ', i, 'freq:', labels.count(i)/total_num)
                left_node = Node()
                right_node = Node()
                left_node.depth = node.depth + 1
                right_node.depth = node.depth + 1

                node.leftChild = left_node
                node.rightChild = right_node
                self.grow_tree(left_node, dataLeft)
                self.grow_tree(right_node, dataRight)
            else:
                node.create_leafNode(patches, labels)
