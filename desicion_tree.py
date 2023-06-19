import pandas as pd
import numpy as np
import copy

df = pd.read_csv("data3.csv")
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
split_margins = {"sepal_length": [5, 7], "sepal_width": [2.5, 3.5], "petal_length": [2.5, 4.5], "petal_width": [0.6, 1.6]}


class Node:

    def __init__(self, records, parent, branch):
        self.attribute = None
        self.records = records
        self.parent = parent
        self.branch = branch
        self.children = []
        self.label = None


def NodeLabel(node):
    species = df["species"]
    class1 = 0
    class2 = 0
    class3 = 0

    for element in node.records:
        if species[element] == "Iris-setosa":
            class1 += 1
        elif species[element] == "Iris-versicolor":
            class2 += 1
        else:
            class3 += 1

    if class1 > class2:
        if class1 > class3:
            return "Iris-setosa"
        else:
            return "Iris-virginica"
    else:
        if class2 > class3:
            return "Iris-versicolor"
        else:
            return "Iris-virginica"


def Split(parent):
    attribute = parent.attribute
    feature = df[attribute]
    boundaries = split_margins[attribute]
    boundaries = [-1] + boundaries + [np.inf]

    children = []
    for i in range(len(boundaries) - 1):
        child_records = []

        for element in parent.records:
            if boundaries[i] < feature[element] <= boundaries[i + 1]:
                child_records.append(element)

        if len(child_records) != 0:
            children.append(Node(child_records, parent, boundaries[i]))

    return children


def GINI(node):
    species = df["species"]
    labels = []
    for element in node.records:
        labels.append(species[element])
    labels = np.array(labels)
    class1 = (labels == "Iris-setosa")
    class2 = (labels == "Iris-versicolor")
    class3 = (labels == "Iris-virginica")
    num_class1 = np.sum(class1, axis=0, keepdims=False)
    num_class2 = np.sum(class2, axis=0, keepdims=False)
    num_class3 = np.sum(class3, axis=0, keepdims=False)
    num_records = len(node.records)
    p1 = num_class1 / num_records
    p2 = num_class2 / num_records
    p3 = num_class3 / num_records
    gini = 1 - (pow(p1, 2) + pow(p2, 2) + pow(p3, 2))
    return gini


def GINI_spilt(parent):
    children = Split(parent)
    gini_split = 0
    for child_node in children:
        gini_split += len(child_node.records) * GINI(child_node)
    gini_split /= len(parent.records)
    return gini_split


def Build_DT(root, unexpanded_features, depth, threshold):
    if depth >= threshold or len(unexpanded_features) == 0:
        root.label = NodeLabel(root)
        root.children = None
        return

    lowest_gini = np.inf
    feature_to_expand = ""
    for feature in unexpanded_features:
        root.attribute = feature
        gini_split = GINI_spilt(root)
        if gini_split <= lowest_gini:
            lowest_gini = gini_split
            feature_to_expand = feature

    root.attribute = feature_to_expand
    root.children = Split(root)

    unexpanded_features_copy = copy.deepcopy(unexpanded_features)
    unexpanded_features_copy.remove(feature_to_expand)

    for child in root.children:
        Build_DT(child, unexpanded_features_copy, depth + 1, threshold)


def Predict(node, record):
    if node.label is not None:
        return node.label

    attribute = node.attribute
    value = record[attribute]
    for child in reversed(node.children):
        if child.branch < value:
            return Predict(child, record)


if __name__ == "__main__":
    average_accuracy = 0

    for i in range(5):
        train_set1 = np.append(np.arange(i * 10), np.arange((i + 1) * 10, 50))
        train_set2 = np.append(np.arange(50, (i * 10) + 50), np.arange(((i + 1) * 10) + 50, 100))
        train_set3 = np.append(np.arange(100, (i * 10) + 100), np.arange(((i + 1) * 10) + 100, 150))
        train_set = np.append(train_set1, train_set2)
        train_set = np.append(train_set, train_set3)

        test_set = np.append(np.arange(i * 10, (i + 1) * 10), np.arange((i * 10) + 50, ((i + 1) * 10) + 50))
        test_set = np.append(test_set, np.arange((i * 10) + 100, ((i + 1) * 10) + 100))

        root_node = Node(train_set, None, None)
        depth_threshold = 2
        Build_DT(root_node, features, 0, depth_threshold)

        true_predictions = 0
        target_feature = df["species"]
        for j in range(len(test_set)):
            record_number = test_set[j]
            prediction = Predict(root_node, df.iloc(0)[record_number])
            if prediction == target_feature[record_number]:
                true_predictions += 1
        accuracy = (true_predictions / len(test_set)) * 100
        print("round", i+1, "accuracy:", accuracy)

        average_accuracy += accuracy

    average_accuracy /= 5
    print("average accuracy:", average_accuracy)
