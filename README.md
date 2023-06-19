# flower-species
In this project, we aim to predict a given flower's species according to its sepal length, sepal width, petal length, and petal width.
There are three different species of flowers in our dataset: Iris setosa, Iris versicolor, and Iris virginica. And there are 150 records in the dataset.

Model deployment:
The model we used for this classification task is Decision Tree (DT).
After doing some data visualizations, we determined the split margins for each feature and wrote them in a dictionary called "split_margins".
As you can see, there are two split margins for each feature which means we divide the data into three groups based on a given feature.
Therefore, each non-leaf node in our tree has got three children.
The criteria for choosing a feature to expand is gini index.
The depth threshold of the tree has been determined 2. This value has been obtained empirically.

Evaluation:
As the data are evenly distributed among classes, we can use the simple accuracy measure which indicates the percentage of true predictions.
Based on the 5-fold-cross-validation testing, the proposed model gives an accuracy of about 95.33%.
