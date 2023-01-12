from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.classification import SVMWithSGD
from pathlib import Path
from utils import evaluate

def Experience(dataset_path=Path('./optimalDataset')):
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    spark = SparkSession.builder.getOrCreate()

    # Get dataset
    df = spark.read.load(str(dataset_path / 'data.parquet'))
    labels = spark.read.csv(str(dataset_path / 'labels.csv'), header=True)
    labels = labels.withColumn("target", labels["target"].cast("tinyint"))
    data = df.join(labels, on='customer_ID', how='left')

    # Split dataset into train set and validation set
    train_data, validation_data = data.randomSplit([0.8, 0.2], seed=42)

    # Convert the training and validation data to RDDs of LabeledPoint objects
    train_rdd = train_data.rdd.map(lambda row: LabeledPoint(row['target'], row[1:-1]))
    validation_rdd = validation_data.rdd.map(lambda row: LabeledPoint(row['target'], row[1:-1]))

    # Get labels from RDD 
    train_labels = train_rdd.map(lambda row: row.label)
    validation_labels = validation_rdd.map(lambda row: row.label)

    # Kmean algorithm
    # Convert the training and validation data to RDDs of LabeledPoint objects
    train_rdd_kmeans = train_rdd.map(lambda row: row.features)
    validation_rdd_kmeans = validation_rdd.map(lambda row: row.features)
    # Build the model (cluster the data)
    kmeans = KMeans.train(train_rdd_kmeans, 2, maxIterations=50, initializationMode="random", seed=0)
    # Predict
    predict = validation_rdd_kmeans.map(lambda ad: kmeans.predict(ad))
    report = evaluate(predict, validation_labels)
    print(f"Kmean algorihtm: {report}")

    # Decision Tree algorithm
    # Train the Decision Tree model on the training data
    dt = DecisionTree.trainClassifier(train_rdd, numClasses=2, categoricalFeaturesInfo = {}, impurity='gini', maxDepth=10, maxBins=32)
    # Predict
    predict = dt.predict(validation_rdd.map(lambda row: row.features))
    report = evaluate(predict, validation_labels)
    print(f"Decision Tree algorihtm: {report}")

    # SVM algorithm
    # Train the model
    svm = SVMWithSGD.train(train_rdd, iterations=100, step=0.1, regParam=0.1)
    # Predict
    predict = svm.predict(validation_rdd.map(lambda row: row.features))
    report = evaluate(predict, validation_labels)
    print(f"Kmean algorihtm: {report}")


if __name__ == "__main__":
    Experience()
