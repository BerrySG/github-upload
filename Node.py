import numpy as np


class Node():
    def __init__(self):

        self.type = 'None'
        self.leftChild = -1
        self.rightChild = -1
        self.feature = {'color': -1, 'pixel_location': [-1, -1], 'th': -1}
        self.depth = 0
        self.probabilities = []
        self.split = []

    # Function to create a new split node
    # provide your implementation
    def create_SplitNode(self, leftchild, rightchild, feature):
        self.type = 'split'
        self.leftChild = leftchild
        self.rightChild = rightchild
        self.feature = feature


    # Function to create a new leaf node
    # provide your implementation
    def create_leafNode(self, labels, classes):
        self.type = 'leaf'
        total_num = len(labels)
        self.probabilities.clear()
        for i in range(classes):
            self.probabilities.append(labels.count(i) / total_num)
        print('leaf created')

    def get_type(self):
        return self.type

    def get_depth(self):
        return self.depth

    def set_depth(self, new_depth):
        self.depth = new_depth
        return True

    def set_split(self, new_split):
        self.split = new_split
        return True

