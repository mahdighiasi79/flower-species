import helper_functions as hf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


if __name__ == "__main__":
    arr = [1, 2, 3, 4]
    arr1 = [np.inf] + arr
    print(arr1)
    # df = pd.read_csv("data3.csv")
    # feature1 = df["petal_length"]
    # feature2 = df["petal_width"]
    # label = df["species"]
    # Iris_setosa1 = []
    # Iris_versicolor1 = []
    # Iris_virginica1 = []
    # Iris_setosa2 = []
    # Iris_versicolor2 = []
    # Iris_virginica2 = []
    # train_set = np.append(np.arange(40), np.arange(50, 90))
    # train_set = np.append(train_set, np.arange(100, 140))
    # test_set = np.append(np.arange(40, 50), np.arange(90, 100))
    # test_set = np.append(test_set, np.arange(140, 150))
    #
    # for i in range(len(train_set)):
    #     record_id = train_set[i]
    #     if label[record_id] == "Iris-setosa":
    #         Iris_setosa1.append(feature1[record_id])
    #         Iris_setosa2.append(feature2[record_id])
    #     elif label[record_id] == "Iris-versicolor":
    #         Iris_versicolor1.append(feature1[record_id])
    #         Iris_versicolor2.append(feature2[record_id])
    #     else:
    #         Iris_virginica1.append(feature1[record_id])
    #         Iris_virginica2.append(feature2[record_id])
    #
    # plt.scatter(Iris_setosa1, Iris_setosa2, color="red")
    # plt.scatter(Iris_versicolor1, Iris_versicolor2, color="black")
    # plt.scatter(Iris_virginica1, Iris_virginica2, color="yellow")
    #
    # plt.show()
